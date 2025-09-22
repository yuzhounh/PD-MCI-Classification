import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

# Set font to Arial for plots
plt.rcParams['font.sans-serif'] = ['Arial']

def load_SBE_results():
    """Load SBE computation results"""
    try:
        # Load detailed results
        with open('8_wrappers_SBE_results.pkl', 'rb') as f:
            detailed_data = pickle.load(f)
        
        # Also load CSV results (as backup)
        results_df = pd.read_csv('8_wrappers_SBE_feature_elimination_order.csv', index_col=0)
        
        print("Successfully loaded SBE results!")
        print(f"Results shape: {results_df.shape}")
        
        return results_df, detailed_data
        
    except FileNotFoundError as e:
        print(f"Error: Could not find results files. Please run 9_wrappers_SBE_compute.py first.")
        print(f"Missing file: {e.filename}")
        return None, None
    except Exception as e:
        print(f"Error loading results: {e}")
        return None, None

def create_individual_model_plots(results_df, models):
    """Create individual feature elimination order plots for each model"""
    plt.style.use('default')
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.ravel()
    
    colors = ['#008BFB', '#FF365E', '#00D084', '#FFB627']
    
    for i, model_name in enumerate(models):
        ax = axes[i]
        
        # Get elimination orders for this model
        model_orders = results_df[model_name].values
        feature_labels = results_df.index.tolist()
        
        # Change Kept features (originally 0) to 12
        modified_orders = model_orders.copy()
        modified_orders[modified_orders == 0] = 12
        
        # Sort by elimination order (ascending, eliminated features at top)
        sort_idx = np.argsort(modified_orders)  # Sort ascending
        sorted_orders = modified_orders[sort_idx]
        sorted_labels = [feature_labels[j] for j in sort_idx]
        
        # Create horizontal bar plot
        y_positions = range(len(sorted_labels))
        
        # Set uniform color for all features
        bar_colors = [colors[i]] * len(sorted_orders)
       
        bars = ax.barh(y_positions, sorted_orders, color=bar_colors, alpha=0.8)
        
        # Customize plot
        ax.set_yticks(y_positions)
        ax.set_yticklabels(sorted_labels, fontsize=10)
        ax.set_xlabel('Elimination Order', fontsize=10)
        ax.set_title(f'{model_name}', fontsize=14, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
        
        # Add order labels on bars
        for j, (bar, order, original_order) in enumerate(zip(bars, sorted_orders, [model_orders[sort_idx[k]] for k in range(len(sort_idx))])):
            width = bar.get_width()
            if original_order == 0:  # Originally kept features
                ax.text(width + 0.1, bar.get_y() + bar.get_height()/2, 
                       '12', ha='left', va='center', fontsize=9)
            else:
                ax.text(width + 0.1, bar.get_y() + bar.get_height()/2, 
                       f'{int(order)}', ha='left', va='center', fontsize=9)
        
        # Set x-axis limit
        ax.set_xlim(0, 13)  # Fixed to 13, as maximum value is 12
    
    plt.tight_layout()
    
    # Save plots
    plt.savefig('8_wrappers_SBE_individual_models.pdf', dpi=300, bbox_inches='tight')
    # plt.savefig('8_wrappers_SBE_individual_models.png', dpi=300, bbox_inches='tight')
    print("Saved individual model plots to: 8_wrappers_SBE_individual_models.pdf/.png")
    plt.show()

def create_overall_summary_plot(results_df):
    """Create overall feature importance summary plot"""
    plt.figure(figsize=(5, 4))
    
    # Sort by elimination order (features with higher elimination order are more important, placed at top)
    summary_df = results_df.copy()
    
    # Change average elimination order of 0 to 12
    modified_avg_order = summary_df['Average_Elimination_Order'].copy()
    modified_avg_order[modified_avg_order == 0] = 12
    summary_df['Modified_Average_Order'] = modified_avg_order
    
    # Sort by modified average elimination order in ascending order
    display_df = summary_df.sort_values('Modified_Average_Order', ascending=True)
    
    y_positions = range(len(display_df))
    
    # Set uniform color for all features
    colors = []
    display_values = []
    for _, row in display_df.iterrows():
        original_order = row['Average_Elimination_Order']
        modified_order = row['Modified_Average_Order']
        colors.append('#008BFB')  # Uniform blue color
        display_values.append(modified_order)
    
    bars = plt.barh(y_positions, display_values, color=colors, alpha=0.8)
    
    plt.yticks(y_positions, display_df.index.tolist(), fontsize=10)
    plt.xlabel('Average Elimination Order', fontsize=10)
    plt.grid(axis='x', alpha=0.3)
    
    # Add labels
    for i, (bar, original_order, modified_order) in enumerate(zip(bars, display_df['Average_Elimination_Order'], display_df['Modified_Average_Order'])):
        width = bar.get_width()
        if original_order == 0:
            plt.text(width + 0.1, bar.get_y() + bar.get_height()/2, 
                    '12.0', ha='left', va='center', fontsize=9)
        else:
            plt.text(width + 0.1, bar.get_y() + bar.get_height()/2, 
                    f'{modified_order:.1f}', ha='left', va='center', fontsize=9)
    
    # Set x-axis limit
    plt.xlim(0, 13)  # Fixed to 13
    
    plt.tight_layout()
    
    # Save plots
    plt.savefig('8_wrappers_SBE_overall_importance.pdf', dpi=300, bbox_inches='tight')
    # plt.savefig('8_wrappers_SBE_overall_importance.png', dpi=300, bbox_inches='tight')
    print("Saved overall importance plot to: 8_wrappers_SBE_overall_importance.pdf/.png")
    plt.show()

def create_heatmap(results_df, models):
    """Create feature elimination order heatmap"""
    plt.figure(figsize=(4, 6))
    
    
    heatmap_data = results_df[models].copy()
    
    # Replace 0 values (Kept features) with 12
    heatmap_data = heatmap_data.replace(0, 12)
    
    # Sort by modified average elimination order (descending)
    modified_avg_order = results_df['Average_Elimination_Order'].copy()
    modified_avg_order[modified_avg_order == 0] = 12
    sort_idx = modified_avg_order.argsort()[::-1]  # Sort descending
    heatmap_data = heatmap_data.iloc[sort_idx]
    
    # Keep original column names
    # heatmap_data.columns remain unchanged
    
    # Create custom colormap, 12 as special green
    from matplotlib.colors import ListedColormap
    import matplotlib.cm as cm
    
    # Create heatmap
    sns.heatmap(heatmap_data, 
               annot=True, 
               cmap='GnBu',  # Reverse colormap, dark colors for important (high values), light colors for unimportant
               fmt='.0f',
               cbar_kws={'label': 'Elimination Order'},
               linewidths=0.5,
               vmin=1, vmax=12)  # Set color range
    
    plt.xticks(rotation=0)
    plt.yticks(rotation=0)
    
    plt.tight_layout()
    
    # Save plots
    plt.savefig('8_wrappers_SBE_heatmap.pdf', dpi=300, bbox_inches='tight')
    # plt.savefig('8_wrappers_SBE_heatmap.png', dpi=300, bbox_inches='tight')
    print("Saved heatmap to: 8_wrappers_SBE_heatmap.pdf/.png")
    plt.show()

def print_detailed_summary(results_df, models):
    """Print detailed analysis summary"""
    eliminated_features_mask = results_df['Average_Elimination_Order'] > 0
    kept_features_mask = results_df['Average_Elimination_Order'] == 0
    
    print("\n" + "="*60)
    print("SBE ANALYSIS SUMMARY")
    print("="*60)
    
    print(f"Total number of features: {len(results_df)}")
    print(f"Features eliminated by at least one algorithm: {eliminated_features_mask.sum()}")
    print(f"Features kept by all algorithms: {kept_features_mask.sum()}")
    
    # Statistics by algorithm
    print(f"\nFeatures eliminated by each algorithm:")
    for model in models:
        eliminated_count = (results_df[model] > 0).sum()
        kept_count = (results_df[model] == 0).sum()
        print(f"  {model}: {eliminated_count} eliminated, {kept_count} kept")
    
    # Consensus analysis
    print(f"\nConsensus analysis:")
    
    # Count how many algorithms eliminated each feature
    elimination_count = (results_df[models] > 0).sum(axis=1)
    for i in range(len(models) + 1):
        count = (elimination_count == i).sum()
        if count > 0:
            if i == 0:
                print(f"  Kept by all algorithms (Order = 12): {count} features")
            else:
                print(f"  Eliminated by {i} algorithm(s): {count} features")
    
    # Most important features (kept features)
    if kept_features_mask.sum() > 0:
        print(f"\nMost important features (kept by all algorithms, Order = 12):")
        kept_features = results_df[kept_features_mask]
        for i, (feature, row) in enumerate(kept_features.iterrows()):
            print(f"  {i+1:2d}. {feature}")
    
    # Next most important features (features eliminated last)
    if eliminated_features_mask.sum() > 0:
        print(f"\nNext most important features (eliminated last, top 10):")
        late_elimination = results_df[eliminated_features_mask].sort_values('Average_Elimination_Order', ascending=False).head(10)
        for i, (feature, row) in enumerate(late_elimination.iterrows()):
            avg_order = row['Average_Elimination_Order']
            eliminated_by = [col for col in models if row[col] > 0]
            print(f"  {i+1:2d}. {feature:<30}: Avg elimination order {avg_order:5.1f}, eliminated by {len(eliminated_by)} algorithm(s)")

def run_SBE_visualization():
    """Run complete SBE visualization analysis"""
    print("Loading SBE results...")
    results_df, detailed_data = load_SBE_results()
    
    if results_df is None:
        return
    
    models = detailed_data['models'] if detailed_data else ['LR', 'Linear SVM', 'RF', 'XGBoost']
    
    print("\n=== Creating Visualizations ===")
    
    # 1. Individual model plots
    print("Creating individual model plots...")
    create_individual_model_plots(results_df, models)
    
    # 2. Overall importance plot
    print("Creating overall importance plot...")
    create_overall_summary_plot(results_df)
    
    # 3. Heatmap
    print("Creating heatmap...")
    create_heatmap(results_df, models)
    
    # 4. Print detailed summary
    print_detailed_summary(results_df, models)
    
    print("\n=== SBE Visualization Completed ===")
    print("All plots have been saved as PDF and PNG files.")

# Run the visualization
if __name__ == "__main__":
    run_SBE_visualization()