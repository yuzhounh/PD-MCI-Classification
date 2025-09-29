import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from utils import load_data, prepare_data, subject_level_cross_validation

def get_embedded_feature_importance(model, X_train, y_train, subjects_train, feature_names):
    """
    Get feature importance using embedded methods
    """
    # Use 10-fold cross validation
    cv_folds = subject_level_cross_validation(X_train, y_train, subjects_train, n_folds=10)
    
    # Store feature importance for each fold
    importance_scores = []
    
    for train_idx, val_idx in cv_folds:
        X_fold_train = X_train[train_idx]
        y_fold_train = y_train[train_idx]
        
        # Train model
        model_copy = model.__class__(**model.get_params())
        model_copy.fit(X_fold_train, y_fold_train)
        
        # Get feature importance
        if hasattr(model_copy, 'feature_importances_'):
            # Random Forest and XGBoost
            importance_scores.append(model_copy.feature_importances_)
        elif hasattr(model_copy, 'coef_'):
            # Logistic Regression and Linear SVM
            importance_scores.append(np.abs(model_copy.coef_[0]))
        else:
            raise ValueError(f"Model {type(model_copy)} doesn't support feature importance extraction")
    
    # Calculate average importance
    mean_importance = np.mean(importance_scores, axis=0)
    return mean_importance

def main():
    print("Starting embedded method feature selection analysis...")
    
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
    
    # Apply feature name mapping
    X_train_mapped = X_train.rename(columns=feature_name_mapping)
    X_test_mapped = X_test.rename(columns=feature_name_mapping)
    
    # Save feature names
    feature_names = X_train_mapped.columns.tolist()
    
    # Convert to numpy arrays
    X_train = X_train_mapped.values
    y_train = y_train.values
    subjects_train = subjects_train.values
    X_test = X_test_mapped.values
    y_test = y_test.values
    
    print(f"Training data shape: {X_train.shape}")
    print(f"Test data shape: {X_test.shape}")
    print(f"Number of features: {len(feature_names)}")
    
    # Define models
    models = {
        'LR': LogisticRegression(random_state=42),
        'SVM': SVC(kernel='linear', probability=True, random_state=42),
        'RF': RandomForestClassifier(random_state=42),
        'XGBoost': XGBClassifier(random_state=42)
    }
    
    print("\nStarting model training and feature importance extraction...")
    
    # Store feature importance for all models
    all_importances = {}
    
    for model_name, model in models.items():
        print(f"\nProcessing {model_name}...")
        
        try:
            # Get feature importance
            importance_scores = get_embedded_feature_importance(
                model, X_train, y_train, subjects_train, feature_names
            )
            all_importances[model_name] = importance_scores
            
            print(f"{model_name} feature importance extraction completed")
            
        except Exception as e:
            print(f"{model_name} processing error: {str(e)}")
            continue
    
    # Create feature importance ranking table
    print("\nCreating feature importance ranking table...")
    
    # Create sorted feature list for each model
    feature_rankings = {}
    
    for model_name, importance_scores in all_importances.items():
        # Sort by importance in descending order
        sorted_indices = np.argsort(importance_scores)[::-1]
        sorted_features = [feature_names[i] for i in sorted_indices]
        sorted_scores = importance_scores[sorted_indices]
        
        feature_rankings[model_name] = {
            'features': sorted_features,
            'scores': sorted_scores,
            'ranks': list(range(1, len(sorted_features) + 1))
        }
    
    # Create comprehensive ranking table
    ranking_df = pd.DataFrame(index=feature_names)
    
    for model_name, ranking_data in feature_rankings.items():
        # Create feature to rank mapping
        feature_to_rank = dict(zip(ranking_data['features'], ranking_data['ranks']))
        ranking_df[f'{model_name}_Rank'] = [feature_to_rank[feat] for feat in feature_names]
        
        # Add importance scores
        feature_to_score = dict(zip(ranking_data['features'], ranking_data['scores']))
        ranking_df[f'{model_name}_Score'] = [feature_to_score[feat] for feat in feature_names]
    
    # Calculate average rank
    rank_columns = [col for col in ranking_df.columns if col.endswith('_Rank')]
    ranking_df['Average_Rank'] = ranking_df[rank_columns].mean(axis=1)
    
    # Sort by average rank
    ranking_df = ranking_df.sort_values('Average_Rank')
    
    # Display ranking results
    print("\nFeature importance ranking table:")
    print("=" * 100)
    
    # Create display table (only show ranks)
    display_df = ranking_df[rank_columns + ['Average_Rank']].copy()
    display_df.columns = [col.replace('_Rank', '') for col in display_df.columns]
    
    print(display_df.round(2))
    
    # Save detailed results
    ranking_df.to_csv('9_embedded_feature_rankings.csv')
    print(f"\nDetailed results saved to '9_embedded_feature_rankings.csv'")
    
    # Display top 10 most important features
    print(f"\nTop 10 most important features (by average rank):")
    print("-" * 50)
    top_features = ranking_df.head(10)
    for i, (feature, row) in enumerate(top_features.iterrows(), 1):
        print(f"{i:2d}. {feature:<20} (Average rank: {row['Average_Rank']:.2f})")
    
    # Display top 5 features for each model separately
    print(f"\nTop 5 important features for each model:")
    print("=" * 80)
    
    for model_name, ranking_data in feature_rankings.items():
        print(f"\n{model_name}:")
        for i, (feature, score) in enumerate(zip(ranking_data['features'][:5], 
                                                 ranking_data['scores'][:5]), 1):
            print(f"  {i}. {feature:<20} (Importance: {score:.4f})")
    
    # Create heatmap data
    heatmap_data = ranking_df[rank_columns].copy()
    heatmap_data.columns = [col.replace('_Rank', '') for col in heatmap_data.columns]
    
    # Save heatmap data
    heatmap_data.to_csv('9_embedded_feature_ranking_heatmap.csv')
    
    print(f"\nAnalysis completed!")
    print(f"- Detailed ranking results: 9_embedded_feature_rankings.csv")
    print(f"- Heatmap data: 9_embedded_feature_ranking_heatmap.csv")

if __name__ == "__main__":
    main()