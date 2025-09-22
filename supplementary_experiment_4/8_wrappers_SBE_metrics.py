# SBE Visualization Script - AUC-PR Line Plot with Number of Remaining Features
# Shows the trend of AUC-PR performance for each algorithm as the number of remaining features changes

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle

# Set font to Arial for plots
plt.rcParams['font.sans-serif'] = ['Arial']

def load_SBE_results():
    """Load SBE computation results"""
    try:
        # Load detailed results
        with open('8_wrappers_SBE_results.pkl', 'rb') as f:
            detailed_data = pickle.load(f)
        
        print("Successfully loaded SBE results!")
        return detailed_data
        
    except FileNotFoundError as e:
        print(f"Error: Could not find results files. Please run 8_wrappers_SBE_compute.py first.")
        print(f"Missing file: {e.filename}")
        return None
    except Exception as e:
        print(f"Error loading results: {e}")
        return None

def create_auc_pr_curve_plot(detailed_data):
    """Create AUC-PR line plot showing performance vs number of remaining features"""
    
    # Set plot style
    plt.style.use('default')
    plt.figure(figsize=(5, 4))
    
    # Colors and marker styles
    colors = ['#008BFB', '#FF6B6B', '#4ECDC4', '#FFD93D']
    markers = ['o', 's', '^', 'D']
    
    # Extract data for each algorithm
    all_results = detailed_data['all_results']
    feature_names = detailed_data['feature_names']
    n_features = len(feature_names)
    
    # Plot curve for each algorithm
    for i, (model_name, model_results) in enumerate(all_results.items()):
        # Get scores_history for this algorithm
        scores_history = model_results['scores_history']
        
        # x-axis: number of remaining features (decreasing from n_features to 1)
        # scores_history[0] corresponds to all features, scores_history[-1] corresponds to fewest features
        x_values = list(range(n_features, n_features - len(scores_history), -1))
        
        # Plot line with markers
        plt.plot(x_values, scores_history, 
                marker=markers[i], 
                color=colors[i], 
                linewidth=2, 
                markersize=6,
                label=model_name,
                alpha=0.8)
    
    # Customize plot
    plt.xlabel('Number of Remaining Features', fontsize=12)
    plt.ylabel('AUC-PR', fontsize=12)
    # plt.title('AUC-PR Performance vs Number of Remaining Features\n(Backward Sequential Elimination)', 
    #           fontsize=14, fontweight='bold')
    # plt.legend(fontsize=11, frameon=True, fancybox=True, shadow=True)
    plt.legend(fontsize=11, frameon=True)
    plt.grid(True, alpha=0.3, linestyle='--')
    
    # Set axis ranges
    plt.xlim(0.5, n_features + 0.5)
    
    # Get range of all AUC-PR values to set y-axis
    all_scores = []
    for model_results in all_results.values():
        all_scores.extend(model_results['scores_history'])
    
    y_min = min(all_scores) * 0.95
    y_max = max(all_scores) * 1.05
    plt.ylim(y_min, y_max)
    
    # Add grid and beautify
    plt.tight_layout()
    
    # Save plot
    plt.savefig('8_wrappers_SBE_auc_pr_curve.pdf', dpi=300, bbox_inches='tight')
    # plt.savefig('8_wrappers_SBE_auc_pr_curve.png', dpi=300, bbox_inches='tight')
    print("Saved AUC-PR curve plot to: 8_wrappers_SBE_auc_pr_curve.pdf/.png")
    plt.show()

def create_detailed_performance_table(detailed_data):
    """Create detailed performance table showing AUC-PR at different numbers of remaining features"""
    
    all_results = detailed_data['all_results']
    feature_names = detailed_data['feature_names']
    n_features = len(feature_names)
    
    
    # Create performance table
    performance_data = []
    
    for model_name, model_results in all_results.items():
        scores_history = model_results['scores_history']
        
        # Get performance at key points
        performance_dict = {
            'Algorithm': model_name,
            'AUC_PR_All_Features': scores_history[0] if len(scores_history) > 0 else 0,
            'AUC_PR_10_Features': scores_history[n_features-10] if len(scores_history) > n_features-10 >= 0 else 0,
            'AUC_PR_5_Features': scores_history[n_features-5] if len(scores_history) > n_features-5 >= 0 else 0,
            'AUC_PR_3_Features': scores_history[n_features-3] if len(scores_history) > n_features-3 >= 0 else 0,
            'AUC_PR_1_Feature': scores_history[-1] if len(scores_history) > 0 else 0,
            'Best_AUC_PR': max(scores_history) if scores_history else 0,
            'Best_at_N_Features': n_features - scores_history.index(max(scores_history)) if scores_history else 0
        }
        
        performance_data.append(performance_dict)
    
    # Convert to DataFrame
    performance_df = pd.DataFrame(performance_data)
    performance_df = performance_df.round(4)
    
    # Save table
    performance_df.to_csv('8_wrappers_SBE_auc_pr_performance_table.csv', index=False)
    print("Saved performance table to: 8_wrappers_SBE_auc_pr_performance_table.csv")
    
    # Print table
    print("\n=== AUC-PR Performance Summary ===")
    print(performance_df.to_string(index=False))
    
    return performance_df

def print_performance_analysis(detailed_data):
    """Print performance analysis summary"""
    
    all_results = detailed_data['all_results']
    feature_names = detailed_data['feature_names']
    n_features = len(feature_names)
    
    print("\n" + "="*60)
    print("SBE AUC-PR PERFORMANCE ANALYSIS")
    print("="*60)
    
    print(f"Total number of features evaluated: {n_features}")
    
    # Analyze performance for each algorithm
    for model_name, model_results in all_results.items():
        scores_history = model_results['scores_history']
        
        if scores_history:
            max_score = max(scores_history)
            max_idx = scores_history.index(max_score)
            initial_score = scores_history[0]  # All features
            final_score = scores_history[-1]   # Fewest features
            best_n_features = n_features - max_idx
            
            print(f"\n{model_name}:")
            print(f"  Initial AUC-PR (all {n_features} features): {initial_score:.4f}")
            print(f"  Best AUC-PR: {max_score:.4f} (with {best_n_features} features)")
            print(f"  Final AUC-PR (1 feature): {final_score:.4f}")
            print(f"  Performance change from all to best: {((max_score - initial_score) / initial_score * 100):+.1f}%")
            print(f"  Performance change from all to 1: {((final_score - initial_score) / initial_score * 100):+.1f}%")
    
    # Find overall best performance
    all_scores = []
    best_models = []
    
    for model_name, model_results in all_results.items():
        scores_history = model_results['scores_history']
        if scores_history:
            max_score = max(scores_history)
            max_idx = scores_history.index(max_score)
            best_n_features = n_features - max_idx
            all_scores.append(max_score)
            best_models.append((model_name, max_score, best_n_features))
    
    if best_models:
        best_models.sort(key=lambda x: x[1], reverse=True)
        print(f"\n=== Overall Best Performance ===")
        for i, (model, score, n_feat) in enumerate(best_models):
            print(f"{i+1}. {model}: {score:.4f} (with {n_feat} features)")

def extract_best_features_for_each_algorithm(detailed_data):
    """Extract features retained by each algorithm at highest AUC-PR"""
    
    all_results = detailed_data['all_results']
    feature_names = detailed_data['feature_names']
    n_features = len(feature_names)
    
    
    print("\n" + "="*80)
    print("OPTIMAL FEATURE SETS FOR EACH ALGORITHM (At Best AUC-PR)")
    print("="*80)
    
    # Store optimal feature sets for all algorithms
    optimal_features_data = []
    
    for model_name, model_results in all_results.items():
        scores_history = model_results['scores_history']
        elimination_order = model_results['elimination_order']
        
        if scores_history:
            # Find best AUC-PR and corresponding number of remaining features
            max_score = max(scores_history)
            best_step = scores_history.index(max_score)
            optimal_n_features = n_features - best_step
            
            # Get optimal feature set (features not eliminated)
            # Features eliminated after best_step should be retained
            if best_step < len(elimination_order):
                eliminated_at_best = set(elimination_order[:best_step])
            else:
                eliminated_at_best = set(elimination_order)
            
            all_feature_indices = set(range(n_features))
            optimal_feature_indices = all_feature_indices - eliminated_at_best
            optimal_feature_names = [feature_names[idx] for idx in sorted(optimal_feature_indices)]
            
            print(f"\n{model_name}:")
            print(f"  Best AUC-PR: {max_score:.4f}")
            print(f"  Optimal number of features: {optimal_n_features}")
            print(f"  Remaining features (alphabetically sorted):")
            
            for i, feature_name in enumerate(optimal_feature_names, 1):
                print(f"    {i:2d}. {feature_name}")
            
            # Store data for later saving
            optimal_features_data.append({
                'Algorithm': model_name,
                'Full_Algorithm_Name': model_name,
                'Best_AUC_PR': max_score,
                'Optimal_N_Features': optimal_n_features,
                'Remaining_Features': optimal_feature_names
            })
    
    # Create and save detailed feature selection table
    save_optimal_features_table(optimal_features_data, feature_names)
    
    return optimal_features_data

def save_optimal_features_table(optimal_features_data, all_feature_names):
    """Save detailed table of optimal feature selections"""
    
    # 1. Save optimal feature list for each algorithm
    optimal_features_summary = []
    for data in optimal_features_data:
        features_str = '; '.join(data['Remaining_Features'])
        optimal_features_summary.append({
            'Algorithm': data['Algorithm'],
            'Best_AUC_PR': data['Best_AUC_PR'],
            'Optimal_N_Features': data['Optimal_N_Features'],
            'Remaining_Features': features_str
        })
    
    summary_df = pd.DataFrame(optimal_features_summary)
    summary_df.to_csv('8_wrappers_SBE_optimal_features_summary.csv', index=False)
    print(f"\nSaved optimal features summary to: 8_wrappers_SBE_optimal_features_summary.csv")
    
    # 2. Create feature retention matrix (whether each feature is retained by each algorithm in optimal state)
    feature_matrix = pd.DataFrame(index=all_feature_names)
    
    for data in optimal_features_data:
        algorithm = data['Algorithm']
        remaining_features = data['Remaining_Features']
        
        # Mark whether each feature is retained (1 for retained, 0 for eliminated)
        feature_matrix[algorithm] = 0
        for feature in remaining_features:
            if feature in feature_matrix.index:
                feature_matrix.loc[feature, algorithm] = 1
    
    # Add summary columns
    feature_matrix['Total_Kept_By'] = feature_matrix.sum(axis=1)
    feature_matrix['Retention_Rate'] = feature_matrix['Total_Kept_By'] / len(optimal_features_data)
    
    # Sort by retention count
    feature_matrix = feature_matrix.sort_values(['Total_Kept_By', 'Retention_Rate'], ascending=[False, False])
    
    feature_matrix.to_csv('8_wrappers_SBE_optimal_features_matrix.csv')
    print(f"Saved optimal features matrix to: 8_wrappers_SBE_optimal_features_matrix.csv")
    
    # 3. Analyze feature consistency
    print(f"\n=== Feature Retention Consensus Analysis ===")
    
    consensus_stats = feature_matrix['Total_Kept_By'].value_counts().sort_index(ascending=False)
    
    for count, num_features in consensus_stats.items():
        if count > 0:
            print(f"  Features kept by {count} algorithm(s): {num_features}")
    
    # Show high consensus features
    high_consensus_features = feature_matrix[feature_matrix['Total_Kept_By'] >= 3]
    if len(high_consensus_features) > 0:
        print(f"\n  High consensus features (kept by ≥3 algorithms):")
        for feature in high_consensus_features.index:
            count = high_consensus_features.loc[feature, 'Total_Kept_By']
            rate = high_consensus_features.loc[feature, 'Retention_Rate']
            print(f"    {feature}: kept by {count}/4 algorithms ({rate:.1%})")
    
    moderate_consensus_features = feature_matrix[feature_matrix['Total_Kept_By'] == 2]
    if len(moderate_consensus_features) > 0:
        print(f"\n  Moderate consensus features (kept by 2 algorithms):")
        for feature in moderate_consensus_features.index:
            print(f"    {feature}")
    
    return feature_matrix

def run_auc_pr_visualization():
    """Run AUC-PR visualization analysis"""
    print("Loading SBE detailed results...")
    detailed_data = load_SBE_results()
    
    if detailed_data is None:
        return
    
    print("\n=== Creating AUC-PR Visualization ===")
    
    # 1. Create AUC-PR curve plot
    print("Creating AUC-PR vs Number of Remaining Features plot...")
    create_auc_pr_curve_plot(detailed_data)
    
    # 2. Create detailed performance table
    print("Creating detailed performance table...")
    create_detailed_performance_table(detailed_data)
    
    # 3. Print performance analysis
    print_performance_analysis(detailed_data)
    
    # 4. Extract optimal feature sets for each algorithm
    print("Extracting optimal feature sets for each algorithm...")
    optimal_features_data = extract_best_features_for_each_algorithm(detailed_data)
    
    print("\n=== AUC-PR Visualization Completed ===")
    print("All plots and tables have been saved.")
    
    return optimal_features_data

# Run the visualization
if __name__ == "__main__":
    run_auc_pr_visualization()