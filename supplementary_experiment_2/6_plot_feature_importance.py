import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
     
# Set font to Arial for plots
plt.rcParams['font.sans-serif'] = ['Arial']

def plot_combined_feature_importance():
    """Plot 4×3 grid comparison chart of feature importance for four algorithms"""
    
    # Read all feature importance files
    lr_data = pd.read_csv('5_LR_feature_importance.csv', index_col=0)
    rf_data = pd.read_csv('5_RF_feature_importance.csv', index_col=0)
    svm_data = pd.read_csv('5_SVM_feature_importance.csv', index_col=0)
    xgb_data = pd.read_csv('5_XGBoost_feature_importance.csv', index_col=0)
    
    # Define algorithm and method correspondence
    algorithms = {
        'LR': {
            'data': lr_data,
            'methods': ['Coefficients', 'SHAP', 'Permutation']
        },
        'SVM': {
            'data': svm_data,
            'methods': ['SVM_Weights', 'SHAP', 'Permutation']
        },
        'RF': {
            'data': rf_data,
            'methods': ['Impurity_Importance', 'SHAP', 'Permutation']
        },
        'XGBoost': {
            'data': xgb_data,
            'methods': ['XGB_Gain', 'SHAP', 'Permutation']
        }
    }
    
    # Method name mapping (for display)
    method_display_names = {
        'Coefficients': 'Coefficients',
        'Impurity_Importance': 'Impurity Importance',
        'SVM_Weights': 'Coefficients', 
        'XGB_Gain': 'Gain',
        'SHAP': 'SHAP',
        'Permutation': 'Permutation'
    }
    
    # Create 4×3 subplots
    fig, axes = plt.subplots(4, 3, figsize=(10, 10))
    
    # Get feature names
    feature_names = lr_data.index.tolist()
    
    # Plot for each algorithm and method combination
    for i, (alg_name, alg_info) in enumerate(algorithms.items()):
        data = alg_info['data']
        methods = alg_info['methods']
        
        for j, method in enumerate(methods):
            ax = axes[i, j]
            
            # Get importance values
            importance_values = data[method].values
            
            # # Normalize XGBoost Gain values for comparison
            # if method == 'XGB_Gain':
            #     importance_values = importance_values / np.sum(np.abs(importance_values))
            
            # Sort by absolute value
            sorted_idx = np.argsort(np.abs(importance_values))
            sorted_features = [feature_names[idx] for idx in sorted_idx]
            sorted_values = importance_values[sorted_idx]
            
            # Plot horizontal bar chart
            bars = ax.barh(range(len(sorted_values)), sorted_values)
            
            # Set y-axis labels
            ax.set_yticks(range(len(sorted_values)))
            ax.set_yticklabels(sorted_features, fontsize=8)
            
            # # Set title and labels
            method_display = method_display_names.get(method, method)
            ax.set_title(f'{alg_name} - {method_display}', fontsize=10, fontweight='bold')
            ax.set_xlabel('Importance', fontsize=9)
            ax.grid(True, alpha=0.3)
            
            # # Set different colors for positive and negative values
            # for k, bar in enumerate(bars):
            #     if sorted_values[k] < 0:
            #         bar.set_color('#FF6B6B')  # Red for negative values
            #     else:
            #         bar.set_color('#4ECDC4')  # Teal for positive values

            # Set different colors for positive and negative values
            for k, bar in enumerate(bars):
                if sorted_values[k] < 0:
                    bar.set_color('#FF0051')  # 255, 0, 81
                else:
                    bar.set_color('#008BFB')  # 0, 139, 251
            
            # # Set x-axis range to ensure different subplots are comparable
            # if method in ['SHAP', 'Permutation']:
            #     ax.set_xlim(-0.1, max(0.6, np.max(np.abs(sorted_values)) * 1.1))
            # elif method == 'Coefficients':
            #     ax.set_xlim(-0.1, max(0.5, np.max(np.abs(sorted_values)) * 1.1))
            # else:
            #     ax.set_xlim(-0.1, np.max(np.abs(sorted_values)) * 1.1)
    
    # Adjust layout
    # plt.tight_layout(pad=3.0)
    plt.tight_layout()
    
    # Save figure
    plt.savefig('6_combined_feature_importance_grid.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('6_combined_feature_importance_grid.svg', dpi=300, bbox_inches='tight')
    
    # # Display figure
    # plt.show()
    
    print("Feature importance 4×3 grid comparison chart has been saved as:")
    print("- 6_combined_feature_importance_grid.pdf")
    # print("- 6_combined_feature_importance_grid.png")

def create_summary_table():
    """Create feature importance summary table"""
    
    # Read all data
    lr_data = pd.read_csv('5_LR_feature_importance.csv', index_col=0)
    rf_data = pd.read_csv('5_RF_feature_importance.csv', index_col=0)
    svm_data = pd.read_csv('5_SVM_feature_importance.csv', index_col=0)
    xgb_data = pd.read_csv('5_XGBoost_feature_importance.csv', index_col=0)
    
    # Create summary table
    summary_data = []
    
    for feature in lr_data.index:
        row = {
            'Feature': feature,
            'LR_Coeff': lr_data.loc[feature, 'Coefficients'],
            'LR_SHAP': lr_data.loc[feature, 'SHAP'],
            'LR_Perm': lr_data.loc[feature, 'Permutation'],
            'SVM_Weights': svm_data.loc[feature, 'SVM_Weights'],
            'SVM_SHAP': svm_data.loc[feature, 'SHAP'],
            'SVM_Perm': svm_data.loc[feature, 'Permutation'],
            'RF_Impurity': rf_data.loc[feature, 'Impurity_Importance'],
            'RF_SHAP': rf_data.loc[feature, 'SHAP'],
            'RF_Perm': rf_data.loc[feature, 'Permutation'],
            'XGB_Gain': xgb_data.loc[feature, 'XGB_Gain'],
            'XGB_SHAP': xgb_data.loc[feature, 'SHAP'],
            'XGB_Perm': xgb_data.loc[feature, 'Permutation']
        }
        summary_data.append(row)
    
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv('6_feature_importance_summary.csv', index=False)
    
    print("\nFeature importance summary table has been saved as: 6_feature_importance_summary.csv")
    print(summary_df.round(4))

if __name__ == "__main__":
    # Plot 4×3 grid chart
    plot_combined_feature_importance()
    
    # Create summary table
    create_summary_table()
