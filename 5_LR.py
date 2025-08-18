import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (roc_auc_score, average_precision_score, confusion_matrix, 
                           accuracy_score, balanced_accuracy_score, precision_score, 
                           recall_score, f1_score, cohen_kappa_score)
import optuna
import shap
from utils import *

# Set font to Arial
plt.rcParams['font.sans-serif'] = ['Arial']

class LRModel:
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.model = None
        self.best_params = None
        self.optimal_threshold = None
        self.trial_count = 0
    
    def create_model(self, params):
        """Create model based on parameters"""
        penalty, solver, C, l1_ratio = params
            
        model_params = {
            'C': C,
            'penalty': penalty,
            'solver': solver,
            'max_iter': 10000,
            'class_weight': 'balanced',
            'random_state': self.random_state
        }
        
        # Only elasticnet needs l1_ratio
        if penalty == 'elasticnet':
            model_params['l1_ratio'] = l1_ratio
            
        return LogisticRegression(**model_params)
    
    def objective_function(self, trial, X, y, subjects, folds, study):
        """Objective function for Optuna Bayesian optimization"""
        self.trial_count += 1
        
        try:
            # Modify parameter sampling strategy to avoid dynamic parameter space issues
            
            # 1. First sample penalty and solver combination
            param_combination = trial.suggest_categorical(
                'param_combination', 
                [
                    'l1_liblinear', 'l1_saga',
                    'l2_lbfgs', 'l2_newton-cg', 'l2_sag', 'l2_saga', 'l2_liblinear',
                    'elasticnet_saga',
                    'none_lbfgs', 'none_newton-cg', 'none_sag', 'none_saga'
                ]
            )
            
            # 2. Parse combination to get penalty and solver
            penalty, solver = param_combination.split('_')
            if penalty == 'none': penalty = None  # Handle none case
            
            # 3. Sample other parameters
            C = trial.suggest_float('C', 0.0001, 1, log=True)
            
            # 4. Decide whether l1_ratio is needed based on penalty
            if penalty == 'elasticnet':
                l1_ratio = trial.suggest_float('l1_ratio', 0.0, 1.0)
            else:
                l1_ratio = 0.5  # Default value, won't be used
            
            params = [penalty, solver, C, l1_ratio]
            
            # Create model
            model = self.create_model(params)
            
            if model is None:
                return 0.0  # Return worst score
            
            # Cross validation
            auc_scores = []
            for train_idx, val_idx in folds:
                X_train_fold, X_val_fold = X[train_idx], X[val_idx]
                y_train_fold, y_val_fold = y[train_idx], y[val_idx]
                
                # Train model
                model.fit(X_train_fold, y_train_fold)
                
                # Predict
                y_val_prob = model.predict_proba(X_val_fold)[:, 1]
                
                # Calculate AUC-PR
                auc_pr = average_precision_score(y_val_fold, y_val_prob)
                auc_scores.append(auc_pr)
            
            mean_auc = np.mean(auc_scores)
            objective_value = mean_auc  # Maximize AUC-PR
            
            return objective_value
                
        except Exception as e:
            print(f"Error in LR objective function: {e}")
            return 0.0
    
    def optimize_hyperparameters(self, X, y, subjects, n_calls=100):
        """Hyperparameter optimization using Optuna Bayesian optimization"""
        print("Starting LR Bayesian optimization (predefined combination strategy)...")
        print("=" * 80)
        
        # Create cross-validation folds
        folds = subject_level_cross_validation(X, y, subjects, n_folds=10, random_state=self.random_state)
        
        # Disable Optuna's default output
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        # Create Optuna study
        study = optuna.create_study(
            direction='maximize',
            sampler=optuna.samplers.TPESampler(seed=self.random_state)
        )
        
        # Reset counter
        self.trial_count = 0
        
        # Custom callback function for output
        def callback(study, trial):
            # Check if trial completed successfully and contains necessary parameters
            if trial.state != optuna.trial.TrialState.COMPLETE or not trial.params:
                return
            
            # Simplified output format
            current_trial = self.trial_count
            total_trials = n_calls
            
            # Safely format parameter display, only show actually used parameters
            params_list = []
            
            if 'param_combination' in trial.params:
                penalty, solver = trial.params['param_combination'].split('_')
                # Handle penalty display value
                display_penalty = 'None' if penalty == 'none' else penalty
                params_list.append(f"penalty={display_penalty}")
                params_list.append(f"solver={solver}")
            
            if 'C' in trial.params:
                params_list.append(f"C={trial.params['C']:.6f}")
            
            # Only show l1_ratio parameter for elasticnet
            if 'l1_ratio' in trial.params and penalty == 'elasticnet':
                params_list.append(f"l1_ratio={trial.params['l1_ratio']:.4f}")
            
            params_str = ", ".join(params_list)
            
            current_auc_pr = trial.value
            best_auc_pr = study.best_value
            best_trial_number = study.best_trial.number + 1
            
            print(f"Trial: {current_trial}/{total_trials}")
            print(f"Parameters: {params_str}")
            print(f"AUC-PR: {current_auc_pr:.6f}")
            print(f"Best AUC-PR (Trial {best_trial_number}): {best_auc_pr:.6f}")
            print()
        
        # Execute Bayesian optimization
        study.optimize(
            lambda trial: self.objective_function(trial, X, y, subjects, folds, study),
            n_trials=n_calls,
            callbacks=[callback]
        )
        
        print("=" * 80)
        print("Search completed")
        
        # Extract optimal parameters
        best_trial = study.best_trial
        
        # Parse optimal parameters
        param_combination = best_trial.params['param_combination']
        penalty, solver = param_combination.split('_')
        
        # Handle none penalty case
        if penalty == 'none':
            penalty = None
        
        self.best_params = {
            'penalty': penalty,
            'solver': solver,
            'C': best_trial.params['C'],
            'l1_ratio': best_trial.params.get('l1_ratio', 0.5)  # Use default value if not present
        }
        
        # Clean optimal parameter display (only show actually used parameters)
        display_penalty = 'None' if penalty is None else penalty
        display_params = {
            'penalty': display_penalty,
            'solver': solver,
            'C': self.best_params['C']
        }
        
        # Only show l1_ratio parameter for elasticnet
        if penalty == 'elasticnet':
            display_params['l1_ratio'] = self.best_params['l1_ratio']
        
        print(f"Optimal parameters: {display_params}")
        print(f"Optimal objective function value: {best_trial.value:.6f} (Optimal AUC-PR: {best_trial.value:.6f})")
        
        return study
    
    def train_final_model(self, X, y, subjects):
        """Train final model using optimal parameters on complete training set"""
        print("Training LR final model...")
        
        # Create final model
        self.model = self.create_model([
            self.best_params['penalty'], 
            self.best_params['solver'],
            self.best_params['C'], 
            self.best_params['l1_ratio']
        ])
        
        # Check if the model was created successfully
        if self.model is None:
            raise ValueError("Model creation failed. Please check the parameters.")
        
        # Train model
        self.model.fit(X, y)
        
        # Calculate optimal threshold - using three different methods
        folds = subject_level_cross_validation(X, y, subjects, n_folds=10, random_state=self.random_state)
        youden_thresholds = []
        f1_thresholds = []
        
        for train_idx, val_idx in folds:
            X_train_fold, X_val_fold = X[train_idx], X[val_idx]
            y_train_fold, y_val_fold = y[train_idx], y[val_idx]
            
            # Train model
            fold_model = self.create_model([
                self.best_params['penalty'], 
                self.best_params['solver'],
                self.best_params['C'], 
                self.best_params['l1_ratio']
            ])
            fold_model.fit(X_train_fold, y_train_fold)
            
            # Predict probabilities
            y_val_prob = fold_model.predict_proba(X_val_fold)[:, 1]
            
            # Calculate two types of optimal thresholds
            youden_thresh = youden_threshold(y_val_fold, y_val_prob)
            f1_thresh = f1_score_threshold(y_val_fold, y_val_prob)
            
            youden_thresholds.append(youden_thresh)
            f1_thresholds.append(f1_thresh)
        
        # Calculate median of three types of optimal thresholds
        self.default = 0.5
        self.f1_median = np.median(f1_thresholds)
        self.youden_median = np.median(youden_thresholds)
        
        # Output comparison of three thresholds
        print("\n" + "="*60)
        print("Optimal threshold calculation results comparison:")
        print("="*60)
        print(f"1. Default threshold:   0.5000")
        print(f"2. F1-score:   {self.f1_median:.4f}")
        print(f"3. Youden index: {self.youden_median:.4f}")
        print("="*60)
        
        # Show detailed threshold information by fold
        print("\nThreshold details by fold:")
        print("Fold\tYouden\tF1-score")
        print("-" * 35)
        for i, (y_thresh, f_thresh) in enumerate(zip(youden_thresholds, f1_thresholds), 1):
            print(f"{i:2d}\t{y_thresh:.4f}\t{f_thresh:.4f}")
        print("-" * 35)
        print(f"Median\t{self.youden_median:.4f}\t{self.f1_median:.4f}")
        print()
    
    def get_cross_validation_performance(self, X, y, subjects):
        """Get cross-validation performance results"""
        print("Calculating training set cross-validation performance...")
        
        # Create cross-validation folds
        folds = subject_level_cross_validation(X, y, subjects, n_folds=10, random_state=self.random_state)
        
        # Store results for each fold
        fold_results = {
            'default': [],  # Default threshold 0.5
            'f1': [],       # F1 optimal threshold
            'youden': []    # Youden optimal threshold
        }
        
        youden_thresholds = []
        f1_thresholds = []
        
        for fold_idx, (train_idx, val_idx) in enumerate(folds):
            X_train_fold, X_val_fold = X[train_idx], X[val_idx]
            y_train_fold, y_val_fold = y[train_idx], y[val_idx]
            
            # Create and train fold model
            fold_model = self.create_model([
                self.best_params['penalty'], 
                self.best_params['solver'], 
                self.best_params['C'], 
                self.best_params['l1_ratio']
            ])
            
            # Train model
            fold_model.fit(X_train_fold, y_train_fold)
            
            # Predict probabilities
            y_val_prob = fold_model.predict_proba(X_val_fold)[:, 1]
            
            # Calculate optimal thresholds
            f1_thresh = f1_score_threshold(y_val_fold, y_val_prob)
            youden_thresh = youden_threshold(y_val_fold, y_val_prob)
            
            f1_thresholds.append(f1_thresh)
            youden_thresholds.append(youden_thresh)
            
            # Calculate predictions and metrics under three thresholds
            thresholds = {
                'default': 0.5, 
                'f1': f1_thresh,
                'youden': youden_thresh
            }
            
            for thresh_type, threshold in thresholds.items():
                y_pred = (y_val_prob >= threshold).astype(int)
                metrics = calculate_all_metrics(y_val_fold, y_pred, y_val_prob, threshold)
                fold_results[thresh_type].append(metrics)
        
        # Calculate mean and standard deviation for each threshold type
        summary_results = {}
        
        for thresh_type in ['default', 'f1', 'youden']:
            metrics_df = pd.DataFrame(fold_results[thresh_type])
            means = metrics_df.mean()
            stds = metrics_df.std()
            
            # Save mean and standard deviation
            summary_results[f'train_{thresh_type}_mean'] = means.to_dict()
            summary_results[f'train_{thresh_type}_std'] = stds.to_dict()
        
        return summary_results
    
    def evaluate_feature_importance(self, X, y, feature_names):
        """Evaluate feature importance"""
        print("Calculating LR feature importance...")
        
        importances = {}
        
        # 1. Weight coefficients
        if hasattr(self.model, 'coef_'):
            importances['Coefficients'] = np.abs(self.model.coef_[0])
        
        # 2. SHAP (LinearExplainer)
        # Use sample subset for SHAP calculation to improve efficiency
        sample_size = min(500, len(X))
        sample_indices = np.random.choice(len(X), sample_size, replace=False)
        X_sample = X[sample_indices]
        
        explainer = shap.LinearExplainer(self.model, X_sample)  
        shap_values = explainer.shap_values(X_sample)
        importances['SHAP'] = np.mean(np.abs(shap_values), axis=0)
        
        # Plot SHAP graph
        plot_shap_summary(shap_values, X_sample, feature_names, 'LR')
        
        # 3. Permutation importance
        perm_importance = get_permutation_importance(self.model, X, y, feature_names, self.random_state)
        importances['Permutation'] = perm_importance
        
        # Plot feature importance comparison graph
        plot_feature_importance_comparison(importances, feature_names, 'LR')
        
        return importances

    def evaluate_and_save_results(self, X_test, y_test):
        """Evaluate test set performance and save results"""
        print("Calculating test set performance...")
        y_prob = self.model.predict_proba(X_test)[:, 1]
        
        # Calculate test set performance under three thresholds
        test_performance = {}
        
        # Default threshold (0.5)
        y_pred_default = (y_prob >= self.default).astype(int)
        test_performance['test_default'] = calculate_all_metrics(y_test, y_pred_default, y_prob, self.default)
        
        # F1 threshold
        y_pred_f1 = (y_prob >= self.f1_median).astype(int)
        test_performance['test_f1'] = calculate_all_metrics(y_test, y_pred_f1, y_prob, self.f1_median)
        
        # Youden threshold
        y_pred_youden = (y_prob >= self.youden_median).astype(int)
        test_performance['test_youden'] = calculate_all_metrics(y_test, y_pred_youden, y_prob, self.youden_median)
        
        return test_performance, y_prob

def main():
    # Set random seed
    np.random.seed(42)
    
    # Load data
    train_data, test_data, feature_name_mapping = load_data(
        'PPMI_7_train.csv', 
        'PPMI_7_test.csv', 
        'PPMI_feature_mapping.csv'
    )
    
    # Prepare training data
    X_train, y_train, subjects_train = prepare_data(train_data)
    X_test, y_test, subjects_test = prepare_data(test_data)
    
    # Convert to numpy arrays
    X_train = X_train.values
    y_train = y_train.values
    subjects_train = subjects_train.values
    X_test = X_test.values
    y_test = y_test.values
    
    # Get feature names (using abbreviations)
    feature_names = [feature_name_mapping.get(col, col) for col in train_data.columns if col not in ['PATNO', 'MCI']]
    
    # Create LR model
    lr_model = LRModel(random_state=42)
    
    # Hyperparameter optimization
    lr_model.optimize_hyperparameters(X_train, y_train, subjects_train)
    
    # Train final model
    lr_model.train_final_model(X_train, y_train, subjects_train)
    
    # Get cross-validation performance
    train_performance = lr_model.get_cross_validation_performance(X_train, y_train, subjects_train)
    
    # Feature importance evaluation
    lr_model.evaluate_feature_importance(X_train, y_train, feature_names)
    
    # Call wrapped function
    test_performance, y_prob = lr_model.evaluate_and_save_results(X_test, y_test)

    # Plot ROC and PR curves
    plot_roc_pr_curves(y_test, y_prob, 'LR')

    # Save training set performance (mean and standard deviation)
    train_df = pd.DataFrame(train_performance)
    train_df.to_csv('5_LR_train_performance.csv')
    print(f"\nTraining set performance results saved to: 5_LR_train_performance.csv")
    
    # Save test set performance
    test_df = pd.DataFrame(test_performance)
    test_df.to_csv('5_LR_test_performance.csv')
    print(f"Test set performance results saved to: 5_LR_test_performance.csv")
    
    # Print results overview
    print("\n" + "="*80)
    print("LR model training set performance results (mean±std):")
    print("="*80)
    print(train_df.round(4))
    
    print("\n" + "="*80)
    print("LR model test set performance results:")
    print("="*80)
    print(test_df.round(4))

if __name__ == "__main__":
    main()