import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set font to Arial for plots
plt.rcParams['font.sans-serif'] = ['Arial']

def plot_feature_importance_heatmap(ranking_file='9_embedded_feature_rankings.csv', 
                                   top_n=20, save_prefix="embedded_"):
    """
    Plot feature importance ranking heatmap
    """
    # Read ranking data
    ranking_df = pd.read_csv(ranking_file, index_col=0)
    
    # Get ranking columns, excluding Average_Rank column
    rank_columns = [col for col in ranking_df.columns if col.endswith('_Rank') and col != 'Average_Rank']
    
    # Select top N features
    top_features = ranking_df.head(top_n)
    heatmap_data = top_features[rank_columns].copy()
    
    # Rename columns, remove _Rank suffix
    heatmap_data.columns = [col.replace('_Rank', '') for col in heatmap_data.columns]
    
    # Create figure
    plt.figure(figsize=(4, 6))
    
    # Draw heatmap (lower rank, darker color)
    ax = sns.heatmap(heatmap_data, 
                     annot=True, 
                     fmt='.0f', 
                     cmap='GnBu_r',  # Reverse color mapping
                     cbar_kws={'label': 'Feature Rank'},
                     linewidths=0.5)
    
    # plt.title(f'Feature Importance Ranking Heatmap (Top {top_n} Features)', 
    #           fontsize=14, fontweight='bold', pad=20)
    # plt.xlabel('Models', fontsize=12)
    # plt.ylabel('Features', fontsize=12)
    
    # Adjust layout
    plt.tight_layout()
    plt.savefig(f'9_{save_prefix}feature_ranking_heatmap.pdf', bbox_inches='tight', dpi=300)
    # plt.savefig(f'9_{save_prefix}feature_ranking_heatmap.png', bbox_inches='tight', dpi=300)
    plt.show()

def plot_average_ranking(ranking_file='9_embedded_feature_rankings.csv', 
                        top_n=15, save_prefix="embedded_"):
    """
    Plot average ranking bar chart
    """
    # Read ranking data
    ranking_df = pd.read_csv(ranking_file, index_col=0)
    
    # Select top N features
    top_features = ranking_df.head(top_n)
    
    # Create figure
    plt.figure(figsize=(5, 4))
    
    # Draw horizontal bar chart
    y_pos = np.arange(len(top_features))
    bars = plt.barh(y_pos, top_features['Average_Rank'], 
                    color='#008BFB', alpha=0.8)
    
    # Set labels
    plt.yticks(y_pos, top_features.index, fontsize=10)
    plt.xlabel('Average Rank', fontsize=10)
    # plt.title(f'Feature Importance - Average Ranking (Top {top_n} Features)', 
    #           fontsize=14, fontweight='bold')
    
    # Set x-axis range to 1.1 times the maximum value
    max_value = top_features['Average_Rank'].max()
    plt.xlim(0, max_value * 1.1)
    
    # Add value labels
    for i, (bar, value) in enumerate(zip(bars, top_features['Average_Rank'])):
        plt.text(value + 0.1, bar.get_y() + bar.get_height()/2, 
                f'{value:.1f}', ha='left', va='center', fontsize=9)
    
    # Invert y-axis so rank 1 is at the top
    plt.gca().invert_yaxis()
    
    # Add grid
    plt.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'9_{save_prefix}average_ranking.pdf', bbox_inches='tight', dpi=300)
    # plt.savefig(f'9_{save_prefix}average_ranking.png', bbox_inches='tight', dpi=300)
    plt.show()

def plot_model_comparison(ranking_file='9_embedded_feature_rankings.csv', 
                         top_n=10, save_prefix="embedded_"):
    """
    Plot feature importance comparison across models
    """
    # Read ranking data
    ranking_df = pd.read_csv(ranking_file, index_col=0)
    
    # Get ranking and score columns
    rank_columns = [col for col in ranking_df.columns if col.endswith('_Rank')]
    score_columns = [col for col in ranking_df.columns if col.endswith('_Score')]
    
    # Select top N features
    top_features = ranking_df.head(top_n)
    
    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.ravel()
    
    # Use specified colors
    colors = ['#008BFB', '#FF365E', '#00D084', '#FFB627']
    
    model_names = [col.replace('_Score', '') for col in score_columns]
    
    for i, (model_name, score_col) in enumerate(zip(model_names, score_columns)):
        ax = axes[i]
        
        # Re-sort by current model's scores
        model_data = ranking_df.sort_values(score_col, ascending=False).head(top_n)
        
        # Draw bar chart
        bars = ax.barh(range(len(model_data)), model_data[score_col], 
                       color=colors[i], alpha=0.8)
        
        # Set labels
        ax.set_yticks(range(len(model_data)))
        ax.set_yticklabels(model_data.index, fontsize=10)
        ax.set_xlabel('Importance', fontsize=10)
        ax.set_title(f'{model_name}', fontsize=14, fontweight='bold')
        
        # Invert y-axis
        ax.invert_yaxis()
        
        # Set x-axis range to 1.1 times the maximum value
        max_value = model_data[score_col].max()
        ax.set_xlim(0, max_value * 1.15)
        
        # Add value labels
        for j, (bar, value) in enumerate(zip(bars, model_data[score_col])):
            # Use scientific notation for SVM, keeping two decimal places
            if model_name == 'SVM':
                value_text = f'{value:.2e}'
            else:
                value_text = f'{value:.3f}'
            
            ax.text(value + max(model_data[score_col]) * 0.01, 
                   bar.get_y() + bar.get_height()/2, 
                   value_text, ha='left', va='center', fontsize=9)
        
        # Add grid
        ax.grid(axis='x', alpha=0.3)
    
    # plt.suptitle(f'Feature Importance Comparison Across Models (Top {top_n} per Model)', 
    #              fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'9_{save_prefix}model_comparison.pdf', bbox_inches='tight', dpi=300)
    # plt.savefig(f'9_{save_prefix}model_comparison.png', bbox_inches='tight', dpi=300)
    plt.show()

def create_summary_table(ranking_file='9_embedded_feature_rankings.csv', 
                        top_n=15, save_prefix="embedded_"):
    """
    Create summary table
    """
    # Read ranking data
    ranking_df = pd.read_csv(ranking_file, index_col=0)
    
    # Get ranking columns
    rank_columns = [col for col in ranking_df.columns if col.endswith('_Rank')]
    
    # Select top N features
    top_features = ranking_df.head(top_n)
    
    # Create summary table
    summary_df = top_features[rank_columns + ['Average_Rank']].copy()
    summary_df.columns = [col.replace('_Rank', '') if col != 'Average_Rank' else col for col in summary_df.columns]
    
    # Add ranking position
    summary_df.insert(0, 'Overall_Rank', range(1, len(summary_df) + 1))
    
    print("=" * 100)
    print(f"Feature Importance Summary Table (Top {top_n} Features)")
    print("=" * 100)
    print(summary_df.round(1).to_string())
    
    # Save table
    summary_df.to_csv(f'9_{save_prefix}summary_table.csv')
    
    return summary_df

def generate_all_plots():
    """
    Generate all plots
    """
    print("Generating feature importance visualization plots...")
    
    try:
        # 1. Heatmap
        print("\n1. Generating feature importance ranking heatmap...")
        plot_feature_importance_heatmap(top_n=20)
        
        # 2. Average ranking plot
        print("2. Generating average ranking bar chart...")
        plot_average_ranking(top_n=15)
        
        # 3. Model comparison plot
        print("3. Generating model comparison plot...")
        plot_model_comparison(top_n=12)
        
        # 4. Summary table
        print("4. Generating summary table...")
        summary_df = create_summary_table(top_n=15)
        
        print("\nAll visualization plots generated successfully!")
        print("Generated files:")
        print("- 9_embedded_feature_ranking_heatmap.pdf/png")
        print("- 9_embedded_average_ranking.pdf/png") 
        print("- 9_embedded_model_comparison.pdf/png")
        print("- 9_embedded_summary_table.csv")
        
        return summary_df
        
    except FileNotFoundError:
        print("Error: Could not find '9_embedded_feature_rankings.csv' file")
        print("Please run the main analysis script first to generate ranking data")
        return None
    except Exception as e:
        print(f"Error occurred while generating plots: {str(e)}")
        return None

if __name__ == "__main__":
    generate_all_plots()