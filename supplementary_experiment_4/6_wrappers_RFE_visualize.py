# Progressive RFE Feature Selection - Visualization Module
# Load computation results and generate all charts

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import pickle

# Set font to Arial for plots
plt.rcParams['font.sans-serif'] = ['Arial']

def load_results():
    """Load computation results from pickle file"""
    try:
        with open('6_wrappers_RFE_results.pkl', 'rb') as f:
            results_data = pickle.load(f)
        
        all_algorithms_results = results_data['all_algorithms_results']
        all_algorithms_detailed = results_data['all_algorithms_detailed']
        feature_names = results_data['feature_names']
        
        print("Successfully loaded computation results from: 6_wrappers_RFE_results.pkl")
        return all_algorithms_results, all_algorithms_detailed, feature_names
        
    except FileNotFoundError:
        print("Error: 6_wrappers_RFE_results.pkl not found!")
        print("Please run 6_wrappers_RFE_compute.py first to generate the results.")
        return None, None, None


def plot_overall_feature_selection_results(all_algorithms_results, feature_names):
    """Plot overall feature selection results (summed across all scenarios)"""
    
    # Calculate grand total for all algorithms combined
    grand_total_counts = defaultdict(int)
    for alg_results in all_algorithms_results.values():
        for feature, count in alg_results.items():
            grand_total_counts[feature] += count
    
    # Prepare data for plotting
    features = list(grand_total_counts.keys())
    counts = list(grand_total_counts.values())
    
    # Add features that were never selected (with count 0)
    never_selected = [f for f in feature_names if f not in features]
    all_features = features + never_selected
    all_counts = counts + [0] * len(never_selected)
    
    # Sort by count
    sorted_data = sorted(zip(all_features, all_counts), key=lambda x: x[1])
    sorted_features, sorted_counts = zip(*sorted_data) if sorted_data else ([], [])
    
    all_features = list(sorted_features)
    all_counts = list(sorted_counts)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(5, 4))
    
    # Create horizontal bar plot
    y_pos = np.arange(len(all_features))
    bars = ax.barh(y_pos, all_counts, color='#008BFB', alpha=0.8)
    
    # Customize plot
    ax.set_yticks(y_pos)
    ax.set_yticklabels(all_features, fontsize=10)
    ax.set_xlabel('Total Selection Count', fontsize=10)
    ax.grid(axis='x', alpha=0.3)
    
    # Set xlim with increased maximum value
    if all_counts:
        max_count = max(all_counts)
        ax.set_xlim(0, max_count * 1.1)  # Increase xlim maximum by 20%
    
    # Add value labels on bars
    for i, (bar, count) in enumerate(zip(bars, all_counts)):
        if count > 0:
            ax.text(count + max(all_counts)*0.01, bar.get_y() + bar.get_height()/2, 
                   str(count), va='center', fontsize=10)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save plot
    plt.savefig('6_wrappers_RFE_overall_results.pdf', dpi=300, bbox_inches='tight')
    # plt.savefig('6_wrappers_RFE_overall_results.png', dpi=300, bbox_inches='tight')
    
    print("Overall feature selection plot saved as:")
    print("- 6_wrappers_RFE_overall_results.pdf")
    # print("- 6_wrappers_RFE_overall_results.png")
    
    plt.show()

def plot_algorithm_comparison(all_algorithms_results, feature_names):
    """Plot comparison across different algorithms"""
    
    # Create 2x2 subplot grid
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.flatten()
    
    # Define colors
    colors = ['#008BFB', '#FF365E', '#00D084', '#FFB627']
    
    for idx, (alg_name, all_selection_counts) in enumerate(all_algorithms_results.items()):
        ax = axes[idx]
        
        # Prepare data for plotting
        features = list(all_selection_counts.keys())
        counts = list(all_selection_counts.values())
        
        # Add features that were never selected (with count 0)
        never_selected = [f for f in feature_names if f not in features]
        all_features = list(features) + never_selected
        all_counts = list(counts) + [0] * len(never_selected)
        
        # Sort by count
        sorted_data = sorted(zip(all_features, all_counts), key=lambda x: x[1])
        sorted_features, sorted_counts = zip(*sorted_data) if sorted_data else ([], [])
        all_features = list(sorted_features)
        all_counts = list(sorted_counts)
        
        # Create horizontal bar plot
        y_pos = np.arange(len(all_features))
        bars = ax.barh(y_pos, all_counts, color=colors[idx], alpha=0.8)
        
        # Customize plot
        ax.set_yticks(y_pos)
        ax.set_yticklabels(all_features, fontsize=10)
        ax.set_xlabel('Total Selection Count', fontsize=11)
        ax.set_title(f'{alg_name}', fontsize=13, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
        
        # Set xlim with increased maximum value
        if all_counts:
            max_count = max(all_counts)
            ax.set_xlim(0, max_count * 1.1)  # Increase xlim maximum by 20%
        
        # Add value labels on bars
        for i, (bar, count) in enumerate(zip(bars, all_counts)):
            if count > 0:
                ax.text(count + max(all_counts)*0.02, bar.get_y() + bar.get_height()/2, 
                       str(count), va='center', fontsize=9)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save plot
    plt.savefig('6_wrappers_RFE_algorithm_comparison.pdf', dpi=300, bbox_inches='tight')
    # plt.savefig('6_wrappers_RFE_algorithm_comparison.png', dpi=300, bbox_inches='tight')
    
    print("Algorithm comparison plots saved as:")
    print("- 6_wrappers_RFE_algorithm_comparison.pdf")
    # print("- 6_wrappers_RFE_algorithm_comparison.png")
    
    plt.show()

def create_heatmap_by_scenario(all_algorithms_detailed, feature_names):
    """Create heatmap showing feature selection across different scenarios"""
    
    # Create data for heatmap
    algorithms = list(all_algorithms_detailed.keys())
    n_features_range = list(range(len(feature_names)-1, 0, -1))  # [11, 10, 9, ..., 1] for 12 features
    
    # Create separate heatmap for each algorithm
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.flatten()
    
    for idx, alg_name in enumerate(algorithms):
        ax = axes[idx]
        
        # Create matrix for current algorithm
        heatmap_data = []
        scenario_labels = []
        
        for n_features in n_features_range:
            if n_features in all_algorithms_detailed[alg_name]:
                selection_counts = all_algorithms_detailed[alg_name][n_features]['selection_counts']
                row_data = [selection_counts.get(feature, 0) for feature in feature_names]
                heatmap_data.append(row_data)
                scenario_labels.append(f"{n_features}")
        
        heatmap_matrix = np.array(heatmap_data)
        
        # Create heatmap
        sns.heatmap(heatmap_matrix, 
                   xticklabels=feature_names,
                   yticklabels=scenario_labels,
                   annot=True, 
                   fmt='d',
                   cmap='GnBu',
                   cbar_kws={'label': 'Selection Count (out of 10 folds)'},
                   ax=ax)
        
        ax.set_title(f'{alg_name}', 
                    fontsize=14, fontweight='bold')
        ax.set_ylabel('Number of Features Selected', fontsize=12)
        
        # Rotate x-axis labels for better readability
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right', fontsize=9)
    
    plt.tight_layout()
    
    # Save heatmap
    plt.savefig('6_wrappers_RFE_heatmap.pdf', dpi=300, bbox_inches='tight')
    # plt.savefig('6_wrappers_RFE_heatmap.png', dpi=300, bbox_inches='tight')
    
    print("Heatmap saved as:")
    print("- 6_wrappers_RFE_heatmap.pdf")
    # print("- 6_wrappers_RFE_heatmap.png")
    
    plt.show()

def print_visualization_summary(all_algorithms_results, feature_names):
    """Print summary of visualization results"""
    
    # Calculate grand total for ranking
    grand_total_counts = defaultdict(int)
    for alg_results in all_algorithms_results.values():
        for feature, count in alg_results.items():
            grand_total_counts[feature] += count
    
    print("\n" + "="*80)
    print("VISUALIZATION SUMMARY")
    print("="*80)
    
    print(f"\nAnalysis visualized for {len(feature_names)} features:")
    print(f"- Scenarios tested: selecting {len(feature_names)-1} down to 1 features")
    print(f"- Cross-validation folds: 10")
    print(f"- Algorithms tested: {len(all_algorithms_results)}")
    
    print(f"\nTop 10 most frequently selected features (across all algorithms and scenarios):")
    sorted_total = sorted(grand_total_counts.items(), key=lambda x: x[1], reverse=True)
    for rank, (feature, total_count) in enumerate(sorted_total[:10], 1):
        total_possible = len(all_algorithms_results) * (len(feature_names) * 10 - 10)  # Maximum possible selections
        percentage = (total_count / total_possible) * 100 if total_possible > 0 else 0
        print(f"  {rank:2d}. {feature}: {total_count} selections ({percentage:.1f}% of total possible)")
    
    print(f"\nGenerated visualization files:")
    print(f"- 6_wrappers_RFE_overall_results.pdf/.png")
    print(f"- 6_wrappers_RFE_algorithm_comparison.pdf/.png")
    print(f"- 6_wrappers_RFE_heatmap.pdf/.png")
    
    print(f"\nVisualization complete!")

def main():
    """Main execution function for visualization"""
    print("Starting Progressive RFE Feature Selection Analysis - VISUALIZATION")
    print("="*60)
    
    # Load computation results
    all_algorithms_results, all_algorithms_detailed, feature_names = load_results()
    
    if all_algorithms_results is None:
        return
    
    
    print(f"Loaded results for {len(feature_names)} features and {len(all_algorithms_results)} algorithms")
    
    # Generate all visualizations
    print("\n" + "="*40)
    print("Generating Overall Feature Selection Plot...")
    plot_overall_feature_selection_results(all_algorithms_results, feature_names)
    
    print("\n" + "="*40)
    print("Generating Algorithm Comparison Plots...")
    plot_algorithm_comparison(all_algorithms_results, feature_names)
    
    print("\n" + "="*40)
    print("Generating Heatmap by Scenario...")
    create_heatmap_by_scenario(all_algorithms_detailed, feature_names)
    
    # Print summary
    print_visualization_summary(all_algorithms_results, feature_names)

if __name__ == "__main__":
    main()
