# SFS Visualization Script - Load computation results and generate charts
# Visualization based on feature selection order

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
        
        # Also load CSV results directly (as backup)
        results_df = pd.read_csv('7_wrappers_SFS_feature_selection_order.csv', index_col=0)
        
        print("Successfully loaded SFS results!")
        print(f"Results shape: {results_df.shape}")
        
        return results_df, detailed_data
        
    except FileNotFoundError as e:
        print(f"Error: Could not find results files. Please run 7_wrappers_SFS_compute.py first.")
        print(f"Missing file: {e.filename}")
        return None, None
    except Exception as e:
        print(f"Error loading results: {e}")
        return None, None

def create_individual_model_plots(results_df, models):
    """Create individual feature selection order plots for each model"""
    plt.style.use('default')
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.ravel()
    
    colors = ['#008BFB', '#FF365E', '#00D084', '#FFB627']
    
    for i, model_name in enumerate(models):
        ax = axes[i]
        
        # Get selection orders for this model
        model_orders = results_df[model_name].values
        feature_labels = results_df.index.tolist()
        
        # Only show selected features (order > 0)
        selected_mask = model_orders > 0
        selected_orders = model_orders[selected_mask]
        selected_labels = [feature_labels[j] for j in range(len(feature_labels)) if selected_mask[j]]
        
        # Sort by selection order (ascending, i.e., top to bottom by order)
        if len(selected_orders) > 0:
            sort_idx = np.argsort(selected_orders)
            selected_orders = selected_orders[sort_idx]
            selected_labels = [selected_labels[j] for j in sort_idx]
            
            # Create horizontal bar plot - directly show order values
            # Reverse y-axis order so features with smaller orders appear at the top
            y_positions = range(len(selected_labels)-1, -1, -1)
            bars = ax.barh(y_positions, selected_orders, color=colors[i], alpha=0.8)
            
            # Customize plot
            ax.set_yticks(y_positions)
            ax.set_yticklabels(selected_labels, fontsize=10)
            ax.set_xlabel('Selection Order', fontsize=10)
            ax.set_title(f'{model_name}', fontsize=14, fontweight='bold')
            ax.grid(axis='x', alpha=0.3)
            
            # Add order labels on bars
            for j, (bar, order) in enumerate(zip(bars, selected_orders)):
                width = bar.get_width()
                ax.text(width + 0.1, bar.get_y() + bar.get_height()/2, 
                       f'{int(order)}', ha='left', va='center', fontsize=9)
            
            # Set x-axis limit
            ax.set_xlim(0, max(selected_orders) * 1.1)
        else:
            ax.text(0.5, 0.5, 'No features selected', transform=ax.transAxes, 
                   ha='center', va='center', fontsize=12)
    
    plt.tight_layout()
    # plt.suptitle('Feature Selection Order by Algorithm (Lower order = Higher importance)', 
    #              fontsize=16, fontweight='bold', y=1.02)
    
    # Save plots
    plt.savefig('7_wrappers_SFS_individual_models.pdf', dpi=300, bbox_inches='tight')
    # plt.savefig('7_wrappers_SFS_individual_models.png', dpi=300, bbox_inches='tight')
    print("Saved individual model plots to: 7_wrappers_SFS_individual_models.pdf/.png")
    plt.show()

def create_overall_summary_plot(results_df):
    """Create overall feature importance summary plot"""
    plt.figure(figsize=(5, 4))
    
    # Only show features selected by at least one algorithm
    selected_features_mask = results_df['Average_Order'] < 999
    if selected_features_mask.sum() > 0:
        summary_df = results_df[selected_features_mask].copy()
        
        # Sort by Average_Order ascending (already sorted in compute script, but ensure correct order)
        summary_df = summary_df.sort_values('Average_Order', ascending=True)
        
        # Reverse y-axis order so features with smaller Average_Order appear at the top
        y_positions = range(len(summary_df)-1, -1, -1)
        bars = plt.barh(y_positions, summary_df['Average_Order'], color='#008BFB', alpha=0.8)
        
        plt.yticks(y_positions, summary_df.index.tolist(), fontsize=10)
        plt.xlabel('Average Selection Order', fontsize=10)
        # plt.title('Overall Feature Importance (Based on Average Selection Order)', 
        #          fontsize=14, fontweight='bold')
        plt.grid(axis='x', alpha=0.3)
        
        # Add average order labels
        for i, (bar, avg_order) in enumerate(zip(bars, summary_df['Average_Order'])):
            width = bar.get_width()
            plt.text(width + 0.1, bar.get_y() + bar.get_height()/2, 
                    f'{avg_order:.1f}', ha='left', va='center', fontsize=9)
        
        # Set x-axis limit
        plt.xlim(0, max(summary_df['Average_Order']) * 1.1)
        
        plt.tight_layout()
        
        # Save plots
        plt.savefig('7_wrappers_SFS_overall_importance.pdf', dpi=300, bbox_inches='tight')
        # plt.savefig('7_wrappers_SFS_overall_importance.png', dpi=300, bbox_inches='tight')
        print("Saved overall importance plot to: 7_wrappers_SFS_overall_importance.pdf/.png")
        plt.show()

def create_heatmap(results_df, models):
    """Create feature selection order heatmap"""
    plt.figure(figsize=(4, 6))
    
    # Algorithm name mapping
    model_name_mapping = {
        'Logistic Regression': 'LR',
        'Linear SVM': 'SVM',
        'Random Forest': 'RF',
        'XGBoost': 'XGBoost'
    }
    
    # Only show features selected by at least one algorithm
    selected_features_mask = results_df['Average_Order'] < 999
    if selected_features_mask.sum() > 0:
        heatmap_data = results_df[selected_features_mask][models].copy()
        
        # Replace 0 values with NaN to display as white in heatmap
        heatmap_data = heatmap_data.replace(0, np.nan)
        
        # Keep original column names
        # heatmap_data.columns remain unchanged
        
        # Create heatmap
        sns.heatmap(heatmap_data, 
                   annot=True, 
                   cmap='GnBu_r',  # Reverse color mapping, light colors for early selection, dark colors for late selection
                   fmt='.0f',
                   cbar_kws={'label': 'Selection Order'},
                   linewidths=0.5)
        
        # plt.title('Feature Selection Order Heatmap\n(Lower values = Earlier selection)', 
        #          fontsize=14, fontweight='bold')
        # plt.xlabel('Algorithm', fontsize=12)
        # plt.ylabel('Features', fontsize=12)
        plt.xticks(rotation=0)
        plt.yticks(rotation=0)
        
        plt.tight_layout()
        
        # Save plots
        plt.savefig('7_wrappers_SFS_heatmap.pdf', dpi=300, bbox_inches='tight')
        # plt.savefig('7_wrappers_SFS_heatmap.png', dpi=300, bbox_inches='tight')
        print("Saved heatmap to: 7_wrappers_SFS_heatmap.pdf/.png")
        plt.show()

def print_detailed_summary(results_df, models):
    """Print detailed analysis summary"""
    selected_features_mask = results_df['Average_Order'] < 999
    
    print("\n" + "="*60)
    print("SFS ANALYSIS SUMMARY")
    print("="*60)
    
    print(f"Total number of features: {len(results_df)}")
    print(f"Features selected by at least one algorithm: {selected_features_mask.sum()}")
    print(f"Features not selected by any algorithm: {(~selected_features_mask).sum()}")
    
    # Statistics by algorithm
    print(f"\nFeatures selected by each algorithm:")
    for model in models:
        selected_count = (results_df[model] > 0).sum()
        print(f"  {model}: {selected_count} features")
    
    # Consensus analysis
    if selected_features_mask.sum() > 0:
        consensus_df = results_df[selected_features_mask].copy()
        selection_count = (consensus_df[models] > 0).sum(axis=1)
        
        print(f"\nConsensus analysis:")
        for i in range(1, len(models) + 1):
            count = (selection_count == i).sum()
            if count > 0:
                print(f"  Selected by {i} algorithm(s): {count} features")
        
        # Top features
        print(f"\nTop 10 most important features (by average selection order):")
        top_features = consensus_df.head(10)
        for i, (feature, row) in enumerate(top_features.iterrows()):
            avg_order = row['Average_Order']
            selected_by = [col for col in models if row[col] > 0]
            print(f"  {i+1:2d}. {feature:<30}: Avg order {avg_order:5.1f}, selected by {len(selected_by)} algorithm(s)")

def run_SFS_visualization():
    """Run complete SFS visualization analysis"""
    print("Loading SFS results...")
    results_df, detailed_data = load_SFS_results()
    
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
    
    print("\n=== SFS Visualization Completed ===")
    print("All plots have been saved as PDF and PNG files.")

# Run the visualization
if __name__ == "__main__":
    run_SFS_visualization()
