# SFS Visualization Script - AUC-PR vs Number of Features Line Plot
# Shows the trend of AUC-PR performance for each algorithm as the number of selected features changes

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

# Set font to Arial for plots
plt.rcParams['font.sans-serif'] = ['Arial']

def load_SFS_results():
    """Load SFS computation results"""
    try:
        # Load detailed results
        with open('7_wrappers_SFS_results.pkl', 'rb') as f:
            detailed_data = pickle.load(f)
        
        print("Successfully loaded SFS results!")
        return detailed_data
        
    except FileNotFoundError as e:
        print(f"Error: Could not find results files. Please run 7_wrappers_SFS_compute.py first.")
        print(f"Missing file: {e.filename}")
        return None
    except Exception as e:
        print(f"Error loading results: {e}")
        return None

def create_auc_pr_curve_plot(detailed_data):
    """Create AUC-PR vs Number of Features line plot"""
    
    
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
        scores_history = model_results[n_features]['scores_history']
        
        # x-axis: number of features (1 to n_features)
        x_values = list(range(1, len(scores_history) + 1))
        
        # Plot line with markers
        plt.plot(x_values, scores_history, 
                marker=markers[i], 
                color=colors[i], 
                linewidth=2, 
                markersize=6,
                label=model_name,
                alpha=0.8)
    
    # Customize plot
    plt.xlabel('Number of Selected Features', fontsize=12)
    plt.ylabel('AUC-PR', fontsize=12)
    # plt.title('AUC-PR Performance vs Number of Selected Features\n(Forward Sequential Selection)', 
    #           fontsize=14, fontweight='bold')
    # plt.legend(fontsize=11, frameon=True, fancybox=True, shadow=True)
    plt.legend(fontsize=11, frameon=True)
    plt.grid(True, alpha=0.3, linestyle='--')
    
    # Set axis ranges
    plt.xlim(0.5, len(feature_names) + 0.5)
    
    # Get range of all AUC-PR values to set y-axis
    all_scores = []
    for model_results in all_results.values():
        all_scores.extend(model_results[n_features]['scores_history'])
    
    y_min = min(all_scores) * 0.95
    y_max = max(all_scores) * 1.05
    plt.ylim(y_min, y_max)
    
    # Add grid and beautify
    plt.tight_layout()
    
    # Save plot
    plt.savefig('7_wrappers_SFS_auc_pr_curve.pdf', dpi=300, bbox_inches='tight')
    # plt.savefig('7_wrappers_SFS_auc_pr_curve.png', dpi=300, bbox_inches='tight')
    print("Saved AUC-PR curve plot to: 7_wrappers_SFS_auc_pr_curve.pdf/.png")
    plt.show()

def create_detailed_performance_table(detailed_data):
    """Create detailed performance table showing AUC-PR at different numbers of features"""
    
    all_results = detailed_data['all_results']
    feature_names = detailed_data['feature_names']
    n_features = len(feature_names)
    
    
    # Create performance table
    performance_data = []
    
    for model_name, model_results in all_results.items():
        scores_history = model_results[n_features]['scores_history']
        
        # Get performance at key points
        performance_dict = {
            'Algorithm': model_name,
            'AUC_PR_1_Feature': scores_history[0] if len(scores_history) > 0 else 0,
            'AUC_PR_5_Features': scores_history[4] if len(scores_history) > 4 else 0,
            'AUC_PR_10_Features': scores_history[9] if len(scores_history) > 9 else 0,
            'AUC_PR_20_Features': scores_history[19] if len(scores_history) > 19 else 0,
            'AUC_PR_All_Features': scores_history[-1] if len(scores_history) > 0 else 0,
            'Best_AUC_PR': max(scores_history) if scores_history else 0,
            'Best_at_N_Features': scores_history.index(max(scores_history)) + 1 if scores_history else 0
        }
        
        performance_data.append(performance_dict)
    
    # Convert to DataFrame
    performance_df = pd.DataFrame(performance_data)
    performance_df = performance_df.round(4)
    
    # Save table
    performance_df.to_csv('7_wrappers_SFS_auc_pr_performance_table.csv', index=False)
    print("Saved performance table to: 7_wrappers_SFS_auc_pr_performance_table.csv")
    
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
    print("SFS AUC-PR PERFORMANCE ANALYSIS")
    print("="*60)
    
    print(f"Total number of features evaluated: {n_features}")
    
    # Analyze performance for each algorithm
    for model_name, model_results in all_results.items():
        scores_history = model_results[n_features]['scores_history']
        
        if scores_history:
            max_score = max(scores_history)
            max_idx = scores_history.index(max_score)
            initial_score = scores_history[0]
            final_score = scores_history[-1]
            
            print(f"\n{model_name}:")
            print(f"  Initial AUC-PR (1 feature): {initial_score:.4f}")
            print(f"  Best AUC-PR: {max_score:.4f} (at {max_idx + 1} features)")
            print(f"  Final AUC-PR (all features): {final_score:.4f}")
            print(f"  Improvement: {((max_score - initial_score) / initial_score * 100):+.1f}%")
    
    # Find overall best performance
    all_scores = []
    best_models = []
    
    for model_name, model_results in all_results.items():
        scores_history = model_results[n_features]['scores_history']
        if scores_history:
            max_score = max(scores_history)
            all_scores.append(max_score)
            best_models.append((model_name, max_score))
    
    if best_models:
        best_models.sort(key=lambda x: x[1], reverse=True)
        print(f"\n=== Overall Best Performance ===")
        for i, (model, score) in enumerate(best_models):
            print(f"{i+1}. {model}: {score:.4f}")

def extract_best_features_for_each_algorithm(detailed_data):
    """Extract features selected by each algorithm at their highest AUC-PR"""
    
    all_results = detailed_data['all_results']
    feature_names = detailed_data['feature_names']
    n_features = len(feature_names)
    
    
    print("\n" + "="*80)
    print("OPTIMAL FEATURE SETS FOR EACH ALGORITHM (At Best AUC-PR)")
    print("="*80)
    
    # Store optimal feature sets for all algorithms
    optimal_features_data = []
    
    for model_name, model_results in all_results.items():
        scores_history = model_results[n_features]['scores_history']
        selected_features = model_results[n_features]['selected_features']
        
        if scores_history:
            # Find best AUC-PR and corresponding number of features
            max_score = max(scores_history)
            optimal_n_features = scores_history.index(max_score) + 1
            
            # Get optimal feature set (first optimal_n_features features)
            optimal_feature_indices = selected_features[:optimal_n_features]
            optimal_feature_names = [feature_names[idx] for idx in optimal_feature_indices]
            
            print(f"\n{model_name}:")
            print(f"  Best AUC-PR: {max_score:.4f}")
            print(f"  Optimal number of features: {optimal_n_features}")
            print(f"  Selected features (in selection order):")
            
            for i, feature_name in enumerate(optimal_feature_names, 1):
                print(f"    {i:2d}. {feature_name}")
            
            # Store data for later saving
            optimal_features_data.append({
                'Algorithm': model_name,
                'Full_Algorithm_Name': model_name,
                'Best_AUC_PR': max_score,
                'Optimal_N_Features': optimal_n_features,
                'Selected_Features': optimal_feature_names
            })
    
    # Create and save detailed feature selection table
    save_optimal_features_table(optimal_features_data, feature_names)
    
    return optimal_features_data

def save_optimal_features_table(optimal_features_data, all_feature_names):
    """Save detailed table of optimal feature selections"""
    
    # 1. Save optimal feature list for each algorithm
    optimal_features_summary = []
    for data in optimal_features_data:
        features_str = '; '.join(data['Selected_Features'])
        optimal_features_summary.append({
            'Algorithm': data['Algorithm'],
            'Best_AUC_PR': data['Best_AUC_PR'],
            'Optimal_N_Features': data['Optimal_N_Features'],
            'Selected_Features': features_str
        })
    
    summary_df = pd.DataFrame(optimal_features_summary)
    summary_df.to_csv('7_wrappers_SFS_optimal_features_summary.csv', index=False)
    print(f"\nSaved optimal features summary to: 7_wrappers_SFS_optimal_features_summary.csv")
    
    # 2. Create feature selection matrix (whether each feature is selected by each algorithm in optimal state)
    feature_matrix = pd.DataFrame(index=all_feature_names)
    
    for data in optimal_features_data:
        algorithm = data['Algorithm']
        selected_features = data['Selected_Features']
        
        # Mark whether each feature is selected (1 for selected, 0 for not selected)
        feature_matrix[algorithm] = 0
        for feature in selected_features:
            if feature in feature_matrix.index:
                feature_matrix.loc[feature, algorithm] = 1
    
    # Add summary columns
    feature_matrix['Total_Selected_By'] = feature_matrix.sum(axis=1)
    feature_matrix['Selection_Rate'] = feature_matrix['Total_Selected_By'] / len(optimal_features_data)
    
    # Sort by selection count
    feature_matrix = feature_matrix.sort_values(['Total_Selected_By', 'Selection_Rate'], ascending=[False, False])
    
    feature_matrix.to_csv('7_wrappers_SFS_optimal_features_matrix.csv')
    print(f"Saved optimal features matrix to: 7_wrappers_SFS_optimal_features_matrix.csv")
    
    # 3. Analyze feature consistency
    print(f"\n=== Feature Selection Consensus Analysis ===")
    
    consensus_stats = feature_matrix['Total_Selected_By'].value_counts().sort_index(ascending=False)
    
    for count, num_features in consensus_stats.items():
        if count > 0:
            print(f"  Features selected by {count} algorithm(s): {num_features}")
    
    # Show high consensus features
    high_consensus_features = feature_matrix[feature_matrix['Total_Selected_By'] >= 3]
    if len(high_consensus_features) > 0:
        print(f"\n  High consensus features (selected by ≥3 algorithms):")
        for feature in high_consensus_features.index:
            count = high_consensus_features.loc[feature, 'Total_Selected_By']
            rate = high_consensus_features.loc[feature, 'Selection_Rate']
            print(f"    {feature}: selected by {count}/4 algorithms ({rate:.1%})")
    
    moderate_consensus_features = feature_matrix[feature_matrix['Total_Selected_By'] == 2]
    if len(moderate_consensus_features) > 0:
        print(f"\n  Moderate consensus features (selected by 2 algorithms):")
        for feature in moderate_consensus_features.index:
            print(f"    {feature}")
    
    return feature_matrix

def run_auc_pr_visualization():
    """Run AUC-PR visualization analysis"""
    print("Loading SFS detailed results...")
    detailed_data = load_SFS_results()
    
    if detailed_data is None:
        return
    
    print("\n=== Creating AUC-PR Visualization ===")
    
    # 1. Create AUC-PR curve plot
    print("Creating AUC-PR vs Number of Features plot...")
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