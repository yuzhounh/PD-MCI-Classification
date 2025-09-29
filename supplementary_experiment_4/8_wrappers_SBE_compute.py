# SBE Computation Script - Execute Sequential Backward Elimination and save results
# Statistical results should count the order of feature elimination, not the number of eliminations

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import average_precision_score
from collections import defaultdict
import pickle
from utils import load_data, prepare_data, subject_level_cross_validation

def sequential_backward_elimination(X, y, subjects, model, n_features_to_keep, cv_folds, scoring='average_precision', feature_names=None):
    """
    Sequential Backward Elimination implementation
    
    Parameters:
    - X: feature matrix
    - y: target vector  
    - subjects: subject IDs for cross-validation
    - model: sklearn model
    - n_features_to_keep: minimum number of features to keep
    - cv_folds: cross-validation folds
    - scoring: scoring metric
    - feature_names: list of feature names (optional)
    
    Returns:
    - elimination_order: list of feature indices in elimination order
    - scores_history: list of scores at each step (before elimination)
    """
    n_total_features = X.shape[1]
    remaining_features = list(range(n_total_features))
    elimination_order = []
    scores_history = []
    
    # Calculate initial performance (all features)
    X_current = X[:, remaining_features]
    scores = []
    for train_idx, val_idx in cv_folds:
        X_train_fold = X_current[train_idx]
        X_val_fold = X_current[val_idx]
        y_train_fold = y[train_idx]
        y_val_fold = y[val_idx]
        
        # Train model
        model_copy = model.__class__(**model.get_params())
        model_copy.fit(X_train_fold, y_train_fold)
        
        # Predict
        if hasattr(model_copy, "predict_proba"):
            y_pred_proba = model_copy.predict_proba(X_val_fold)[:, 1]
        else:
            y_pred_proba = model_copy.decision_function(X_val_fold)
        
        # Calculate AUC-PR
        score = average_precision_score(y_val_fold, y_pred_proba)
        scores.append(score)
    
    initial_score = np.mean(scores)
    scores_history.append(initial_score)
    
    feature_display = f"All {n_total_features} features"
    print(f"Initial: {feature_display}, Score: {initial_score:.4f}")
    
    # Start elimination process
    n_features_to_eliminate = n_total_features - n_features_to_keep
    
    for step in range(n_features_to_eliminate):
        best_score = -1
        worst_feature = None
        
        # Try removing each remaining feature, find the one with minimal performance drop
        for feature in remaining_features:
            # Create feature subset after removing current feature
            temp_features = [f for f in remaining_features if f != feature]
            X_subset = X[:, temp_features]
            
            # Perform cross-validation
            scores = []
            for train_idx, val_idx in cv_folds:
                X_train_fold = X_subset[train_idx]
                X_val_fold = X_subset[val_idx]
                y_train_fold = y[train_idx]
                y_val_fold = y[val_idx]
                
                # Train model
                model_copy = model.__class__(**model.get_params())
                model_copy.fit(X_train_fold, y_train_fold)
                
                # Predict
                if hasattr(model_copy, "predict_proba"):
                    y_pred_proba = model_copy.predict_proba(X_val_fold)[:, 1]
                else:
                    y_pred_proba = model_copy.decision_function(X_val_fold)
                
                # Calculate AUC-PR
                score = average_precision_score(y_val_fold, y_pred_proba)
                scores.append(score)
            
            avg_score = np.mean(scores)
            
            # Select the feature that gives best performance after removal (least impact on performance)
            if avg_score > best_score:
                best_score = avg_score
                worst_feature = feature
        
        # Remove worst feature
        elimination_order.append(worst_feature)
        remaining_features.remove(worst_feature)
        scores_history.append(best_score)
        
        # Use feature name if available, otherwise use index
        feature_display = feature_names[worst_feature] if feature_names else worst_feature
        print(f"Step {step + 1}: Eliminated feature {feature_display}, Remaining score: {best_score:.4f}")
    
    return elimination_order, scores_history

def run_SBE_computation():
    """Execute SBE computation and save results"""
    
    # Load data
    print("Loading data...")
    train_data, test_data, feature_name_mapping = load_data(
        'PPMI_6_train.csv',
        'PPMI_6_test.csv', 
        '../PPMI_feature_mapping.csv'
    )
    
    # Prepare training data
    X_train, y_train, subjects_train = prepare_data(train_data)
    
    # Prepare test data
    X_test, y_test, subjects_test = prepare_data(test_data)
    
    # Apply feature name mapping before converting to numpy
    X_train_mapped = X_train.rename(columns=feature_name_mapping)
    X_test_mapped = X_test.rename(columns=feature_name_mapping)
    
    # Store feature names before converting to numpy
    feature_names = X_train_mapped.columns.tolist()
    
    # Convert to numpy arrays
    X_train = X_train_mapped.values
    y_train = y_train.values
    subjects_train = subjects_train.values
    X_test = X_test_mapped.values
    y_test = y_test.values
    
    # Get cross-validation folds
    cv_folds = subject_level_cross_validation(X_train, y_train, subjects_train, n_folds=10)
    
    # Define models
    models = {
        'LR': LogisticRegression(random_state=42),
        'SVM': SVC(kernel='linear', probability=True, random_state=42),
        'RF': RandomForestClassifier(random_state=42),
        'XGBoost': XGBClassifier(random_state=42)
    }
    
    # Store results - Modified to store elimination order
    feature_elimination_order = defaultdict(dict)
    all_results = {}
    
    n_features = len(feature_names)  # Total number of features
    n_features_to_keep = 1  # Keep at least 1 feature
    
    print(f"Starting SBE analysis with {n_features} features...")
    
    # Run SBE for each model
    for model_name, model in models.items():
        print(f"\n=== Processing {model_name} ===")
        all_results[model_name] = {}
        
        elimination_order, scores_history = sequential_backward_elimination(
            X_train, y_train, subjects_train, model, n_features_to_keep, cv_folds, feature_names=feature_names
        )
        
        # Record the elimination order for each feature (1 = first eliminated, 2 = second eliminated, etc.)
        for order, feature_idx in enumerate(elimination_order):
            feature_elimination_order[model_name][feature_idx] = order + 1
        
        # For features not eliminated (i.e., kept features), set to 0
        eliminated_features = set(elimination_order)
        all_features = set(range(n_features))
        remaining_features = all_features - eliminated_features
        for feature_idx in remaining_features:
            feature_elimination_order[model_name][feature_idx] = 0
        
        all_results[model_name] = {
            'elimination_order': elimination_order,
            'eliminated_feature_names': [feature_names[i] for i in elimination_order],
            'remaining_features': list(remaining_features),
            'remaining_feature_names': [feature_names[i] for i in remaining_features],
            'scores_history': scores_history
        }
    
    # Create results table - Modified to show elimination order
    print("\n=== Creating Feature Elimination Results ===")
    results_df = pd.DataFrame(index=feature_names)
    
    for model_name in models.keys():
        # If feature was not eliminated, fill with 0; if eliminated, fill with elimination order
        model_order = [feature_elimination_order[model_name].get(i, 0) for i in range(len(feature_names))]
        results_df[model_name] = model_order
    
    # Calculate average elimination order (ignore features not eliminated, i.e., values of 0)
    def calculate_avg_elimination_order(row):
        non_zero_values = [val for val in row if val > 0]
        return np.mean(non_zero_values) if non_zero_values else 0  # Features not eliminated set to 0
    
    results_df['Average_Elimination_Order'] = results_df.apply(calculate_avg_elimination_order, axis=1)
    
    # Sort by average elimination order (higher values = less important, eliminated earlier)
    results_df = results_df.sort_values('Average_Elimination_Order', ascending=False)
    
    print("\nFeature Elimination Order Table (1=first eliminated, 0=not eliminated/kept):")
    print(results_df)
    
    # Save results
    print("\n=== Saving Results ===")
    
    # Save CSV format results table
    results_df.to_csv('8_wrappers_SBE_feature_elimination_order.csv')
    print("Saved feature elimination order table to: 8_wrappers_SBE_feature_elimination_order.csv")
    
    # Save detailed results to pickle file
    detailed_results_data = {
        'results_df': results_df,
        'all_results': all_results,
        'feature_names': feature_names,
        'models': list(models.keys()),
        'feature_elimination_order': dict(feature_elimination_order)
    }
    
    with open('8_wrappers_SBE_results.pkl', 'wb') as f:
        pickle.dump(detailed_results_data, f)
    print("Saved detailed results to: 8_wrappers_SBE_results.pkl")
    
    # Print summary statistics
    eliminated_features_mask = results_df['Average_Elimination_Order'] > 0
    print(f"\n=== Summary Statistics ===")
    print(f"Total number of features: {len(feature_names)}")
    print(f"Number of features eliminated by at least one algorithm: {eliminated_features_mask.sum()}")
    print(f"Number of features kept by all algorithms: {(~eliminated_features_mask).sum()}")
    
    print(f"\nTop 5 most important features (never eliminated or eliminated last):")
    important_features = results_df[~eliminated_features_mask]
    if len(important_features) > 0:
        print("Features never eliminated:")
        for i, (feature, row) in enumerate(important_features.iterrows()):
            kept_by = [col for col in models.keys() if row[col] == 0]
            print(f"{i+1}. {feature}: kept by {kept_by}")
    else:
        # If all features were eliminated by some algorithm, show features eliminated last
        late_elimination = results_df.tail(5)
        print("Features eliminated last (most important):")
        for i, (feature, row) in enumerate(late_elimination.iterrows()):
            avg_order = row['Average_Elimination_Order']
            eliminated_by = [col for col in models.keys() if row[col] > 0]
            print(f"{i+1}. {feature}: Average elimination order {avg_order:.1f}, eliminated by {eliminated_by}")
    
    print("\n=== SBE Computation Completed ===")
    return results_df, all_results

# Run the computation
if __name__ == "__main__":
    results_df, detailed_results = run_SBE_computation()
