# Progressive RFE Feature Selection - Computation Module
# Execute Progressive Recursive Feature Elimination analysis and save results

import pandas as pd
import numpy as np
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from collections import defaultdict
from xgboost import XGBClassifier
import pickle
from utils import load_data, prepare_data, subject_level_cross_validation

def progressive_rfe_feature_selection(X, y, subjects, feature_names, estimator, n_folds=10):
    """
    Perform Progressive Recursive Feature Elimination with cross-validation
    Progressively remove features: start with n-1 features, gradually reduce to only 1 feature
    
    Parameters:
    -----------
    X : numpy.ndarray
        Feature matrix
    y : numpy.ndarray
        Target variable
    subjects : numpy.ndarray
        Subject IDs
    feature_names : list
        List of feature names
    estimator : sklearn estimator
        Base estimator for RFE
    n_folds : int
        Number of cross-validation folds
    
    Returns:
    --------
    all_selection_counts : dict
        Count of how many times each feature was selected across all scenarios and folds
    detailed_results : dict
        Detailed results for each number of features to select
    """
    
    n_features = len(feature_names)
    
    # Get cross-validation folds
    folds = subject_level_cross_validation(X, y, subjects, n_folds=n_folds)
    
    # Initialize counters
    all_selection_counts = defaultdict(int)
    detailed_results = {}
    
    # Start with n-1 features, gradually reduce to only 1 feature
    for n_features_to_select in range(n_features-1, 0, -1):
        print(f"\n{'='*50}")
        print(f"Selecting {n_features_to_select} out of {n_features} features")
        print(f"{'='*50}")
        
        # Initialize counters for current scenario
        current_selection_counts = defaultdict(int)
        feature_rankings = []
        
        # Perform RFE for each fold
        for fold_idx, (train_idx, val_idx) in enumerate(folds):
            print(f"Processing fold {fold_idx + 1}/{n_folds} (selecting {n_features_to_select} features)...")
            
            # Split data
            X_train_fold = X[train_idx]
            y_train_fold = y[train_idx]
            
            # Perform RFE
            rfe = RFE(estimator=estimator, n_features_to_select=n_features_to_select)
            rfe.fit(X_train_fold, y_train_fold)
            
            # Get selected features
            selected_features = [feature_names[i] for i in range(len(feature_names)) if rfe.support_[i]]
            
            # Count selected features for current scenario
            for feature in selected_features:
                current_selection_counts[feature] += 1
                all_selection_counts[feature] += 1  # Add to overall count
            
            # Store feature rankings
            feature_ranking = dict(zip(feature_names, rfe.ranking_))
            feature_rankings.append(feature_ranking)
            
            print(f"Fold {fold_idx + 1} selected features: {selected_features}")
        
        # Store detailed results for current scenario
        detailed_results[n_features_to_select] = {
            'selection_counts': dict(current_selection_counts),
            'feature_rankings': feature_rankings
        }
        
        # Print summary for current scenario
        print(f"\nSummary for selecting {n_features_to_select} features:")
        sorted_features = sorted(current_selection_counts.items(), key=lambda x: x[1], reverse=True)
        for feature, count in sorted_features:
            print(f"  {feature}: selected {count}/{n_folds} times")
    
    return dict(all_selection_counts), detailed_results

def wrapper_methods_analysis():
    """Main function to perform wrapper methods analysis"""
    
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
    
    print(f"Training data shape: {X_train.shape}")
    print(f"Number of subjects: {len(np.unique(subjects_train))}")
    print(f"Target distribution: {np.bincount(y_train.astype(int))}")
    print(f"Total number of features: {len(feature_names)}")
    
    # Define models
    algorithms = {
        'LR': LogisticRegression(random_state=42),
        'SVM': SVC(kernel='linear', probability=True, random_state=42),
        'RF': RandomForestClassifier(random_state=42),
        'XGBoost': XGBClassifier(random_state=42)
    }
    
    # Store results
    all_algorithms_results = {}
    all_algorithms_detailed = {}
    
    # Perform Progressive RFE for each algorithm
    for alg_name, estimator in algorithms.items():
        print(f"\n{'='*60}")
        print(f"Running Progressive RFE with {alg_name}")
        print(f"{'='*60}")
        
        # Perform Progressive RFE
        all_selection_counts, detailed_results = progressive_rfe_feature_selection(
            X_train, y_train, subjects_train, feature_names, estimator
        )
        
        all_algorithms_results[alg_name] = all_selection_counts
        all_algorithms_detailed[alg_name] = detailed_results
        
        # Print overall results for this algorithm
        print(f"\nOverall feature selection results for {alg_name}:")
        sorted_features = sorted(all_selection_counts.items(), key=lambda x: x[1], reverse=True)
        for feature, count in sorted_features:
            total_possible_selections = len(feature_names) * 10 - 10  # Total possible selections
            print(f"  {feature}: selected {count} times (out of {total_possible_selections} possible)")
    
    return all_algorithms_results, all_algorithms_detailed, feature_names

def create_comprehensive_results_table(all_algorithms_results, all_algorithms_detailed, feature_names):
    """Create comprehensive table showing all results"""
    
    # Create overall summary table
    results_df = pd.DataFrame(index=feature_names)
    
    # Add columns for each algorithm (overall counts)
    for alg_name, all_selection_counts in all_algorithms_results.items():
        # Create column with selection counts (0 if not selected)
        selection_counts = [all_selection_counts.get(feature, 0) for feature in feature_names]
        results_df[alg_name] = selection_counts
    
    # Calculate total across all algorithms
    results_df['Total_All_Algorithms'] = results_df.sum(axis=1)
    
    # Sort by total selection count across all algorithms
    results_df = results_df.sort_values('Total_All_Algorithms', ascending=False)
    
    # Save overall results
    results_df.to_csv('6_wrappers_RFE_overall_results.csv')
    print("\nOverall feature selection results saved as: 6_wrappers_RFE_overall_results.csv")
    
    # Create detailed breakdown by number of features
    detailed_breakdown = []
    for alg_name, detailed_results in all_algorithms_detailed.items():
        for n_features_selected, scenario_data in detailed_results.items():
            selection_counts = scenario_data['selection_counts']
            for feature in feature_names:
                count = selection_counts.get(feature, 0)
                detailed_breakdown.append({
                    'Algorithm': alg_name,
                    'Features_Selected': n_features_selected,
                    'Feature': feature,
                    'Selection_Count': count
                })
    
    detailed_df = pd.DataFrame(detailed_breakdown)
    detailed_df.to_csv('6_wrappers_RFE_detailed_breakdown.csv', index=False)
    print("Detailed breakdown saved as: 6_wrappers_RFE_detailed_breakdown.csv")
    
    return results_df, detailed_df

def save_results_for_visualization(all_algorithms_results, all_algorithms_detailed, feature_names):
    """Save results data for visualization scripts"""
    
    results_data = {
        'all_algorithms_results': all_algorithms_results,
        'all_algorithms_detailed': all_algorithms_detailed,
        'feature_names': feature_names
    }
    
    # Save as pickle file
    with open('6_wrappers_RFE_results.pkl', 'wb') as f:
        pickle.dump(results_data, f)
    
    print("Results data saved as: 6_wrappers_RFE_results.pkl")

def main():
    """Main execution function"""
    print("Starting Progressive RFE Feature Selection Analysis - COMPUTATION")
    print("="*60)
    
    # Perform progressive wrapper methods analysis
    all_algorithms_results, all_algorithms_detailed, feature_names = wrapper_methods_analysis()
    
    # Create comprehensive results tables
    results_table, detailed_breakdown = create_comprehensive_results_table(
        all_algorithms_results, all_algorithms_detailed, feature_names
    )
    
    # Save results for visualization
    save_results_for_visualization(all_algorithms_results, all_algorithms_detailed, feature_names)
    
    # Calculate grand total for ranking
    grand_total_counts = defaultdict(int)
    for alg_results in all_algorithms_results.values():
        for feature, count in alg_results.items():
            grand_total_counts[feature] += count
    
    # Print comprehensive summary
    print("\n" + "="*80)
    print("COMPUTATION SUMMARY")
    print("="*80)
    
    print(f"\nAnalysis completed for {len(feature_names)} features:")
    print(f"- Scenarios tested: selecting {len(feature_names)-1} down to 1 features")
    print(f"- Cross-validation folds: 10")
    print(f"- Algorithms tested: {len(all_algorithms_results)}")
    
    print(f"\nTop 10 most frequently selected features (across all algorithms and scenarios):")
    sorted_total = sorted(grand_total_counts.items(), key=lambda x: x[1], reverse=True)
    for rank, (feature, total_count) in enumerate(sorted_total[:10], 1):
        total_possible = len(all_algorithms_results) * (len(feature_names) * 10 - 10)  # Maximum possible selections
        percentage = (total_count / total_possible) * 100 if total_possible > 0 else 0
        print(f"  {rank:2d}. {feature}: {total_count} selections ({percentage:.1f}% of total possible)")
    
    print(f"\nGenerated files:")
    print(f"- 6_wrappers_RFE_overall_results.csv")
    print(f"- 6_wrappers_RFE_detailed_breakdown.csv")
    print(f"- 6_wrappers_RFE_results.pkl (for visualization)")
    
    print(f"\nComputation complete! Run 6_wrappers_5_visualize.py to generate plots.")

if __name__ == "__main__":
    main()
