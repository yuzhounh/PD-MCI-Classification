# SFS Computation Script - Execute Sequential Forward Selection and save results
# Statistical results should track the order in which features are selected, not the frequency of selection

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

def sequential_forward_selection(X, y, subjects, model, n_features, cv_folds, scoring='average_precision', feature_names=None):
    """
    Sequential Forward Selection implementation
    
    Parameters:
    - X: feature matrix
    - y: target vector  
    - subjects: subject IDs for cross-validation
    - model: sklearn model
    - n_features: number of features to select
    - cv_folds: cross-validation folds
    - scoring: scoring metric
    - feature_names: list of feature names (optional)
    
    Returns:
    - selected_features: list of selected feature indices
    - scores_history: list of scores at each step
    """
    n_total_features = X.shape[1]
    selected_features = []
    remaining_features = list(range(n_total_features))
    scores_history = []
    
    for step in range(n_features):
        best_score = -1
        best_feature = None
        
        for feature in remaining_features:
            # Create feature subset
            current_features = selected_features + [feature]
            X_subset = X[:, current_features]
            
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
            
            if avg_score > best_score:
                best_score = avg_score
                best_feature = feature
        
        # Add best feature
        selected_features.append(best_feature)
        remaining_features.remove(best_feature)
        scores_history.append(best_score)
        
        # Use feature name if available, otherwise use index
        feature_display = feature_names[best_feature] if feature_names else best_feature
        print(f"Step {step + 1}: Selected feature {feature_display}, Score: {best_score:.4f}")
    
    return selected_features, scores_history

def run_SFS_computation():
    """Execute SFS computation and save results"""
    
    # Load data
    print("Loading data...")
    train_data, test_data, feature_name_mapping = load_data(
        'PPMI_6_train.csv',
        'PPMI_6_test.csv', 
        'PPMI_feature_mapping.csv'
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
    
    # Store results - Modified to store selection order
    feature_selection_order = defaultdict(dict)
    all_results = {}
    
    n_features = len(feature_names)  # Total number of features
    
    print(f"Starting SFS analysis with {n_features} features...")
    
    # Run SFS for each model and each number of features
    for model_name, model in models.items():
        print(f"\n=== Processing {model_name} ===")
        all_results[model_name] = {}
        
        selected_features, scores_history = sequential_forward_selection(
            X_train, y_train, subjects_train, model, n_features, cv_folds, feature_names=feature_names
        )
        
        # Record the order in which each feature was selected (1 for first selected, 2 for second, etc.)
        for order, feature_idx in enumerate(selected_features):
            feature_selection_order[model_name][feature_idx] = order + 1
        
        all_results[model_name][n_features] = {
            'selected_features': selected_features,
            'feature_names': [feature_names[i] for i in selected_features],
            'scores_history': scores_history
        }
    
    # Create results table - Modified to show selection order
    print("\n=== Creating Feature Selection Results ===")
    results_df = pd.DataFrame(index=feature_names)
    
    for model_name in models.keys():
        # If feature was not selected, fill with 0; if selected, fill with selection order
        model_order = [feature_selection_order[model_name].get(i, 0) for i in range(len(feature_names))]
        results_df[model_name] = model_order
    
    # Calculate average selection order (ignore unselected features, i.e., features with value 0)
    def calculate_avg_order(row):
        non_zero_values = [val for val in row if val > 0]
        return np.mean(non_zero_values) if non_zero_values else 999  # Set unselected features to 999
    
    results_df['Average_Order'] = results_df.apply(calculate_avg_order, axis=1)
    
    # Sort by average selection order (smaller values are more important)
    results_df = results_df.sort_values('Average_Order', ascending=True)
    
    print("\nFeature Selection Order Table (1=first selected, 0=not selected):")
    print(results_df)
    
    # Save results
    print("\n=== Saving Results ===")
    
    # Save CSV format results table
    results_df.to_csv('7_wrappers_SFS_feature_selection_order.csv')
    print("Saved feature selection order table to: 7_wrappers_SFS_feature_selection_order.csv")
    
    # Save detailed results to pickle file
    detailed_results_data = {
        'results_df': results_df,
        'all_results': all_results,
        'feature_names': feature_names,
        'models': list(models.keys()),
        'feature_selection_order': dict(feature_selection_order)
    }
    
    with open('7_wrappers_SFS_results.pkl', 'wb') as f:
        pickle.dump(detailed_results_data, f)
    print("Saved detailed results to: 7_wrappers_SFS_results.pkl")
    
    # Print summary statistics
    selected_features_mask = results_df['Average_Order'] < 999
    print(f"\n=== Summary Statistics ===")
    print(f"Total number of features: {len(feature_names)}")
    print(f"Number of features selected by at least one algorithm: {selected_features_mask.sum()}")
    
    print(f"\nTop 5 most important features (by average selection order):")
    top_features = results_df[selected_features_mask].head()
    for i, (feature, row) in enumerate(top_features.iterrows()):
        avg_order = row['Average_Order']
        selected_by = [col for col in models.keys() if row[col] > 0]
        print(f"{i+1}. {feature}: Average order {avg_order:.1f}, selected by {selected_by}")
    
    print("\n=== SFS Computation Completed ===")
    return results_df, all_results

# Run the computation
if __name__ == "__main__":
    results_df, detailed_results = run_SFS_computation()
