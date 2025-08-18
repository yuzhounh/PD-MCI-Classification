import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, average_precision_score

# Set font to Arial
plt.rcParams['font.sans-serif'] = ['Arial']

# Define algorithm names and corresponding colors
algorithms = ['LR', 'SVM', 'RF', 'XGBoost']
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']  # Blue, Orange, Green, Red
algorithm_names = {
    'LR': 'Logistic Regression',
    'RF': 'Random Forest',
    'SVM': 'Support Vector Machine',
    'XGBoost': 'XGBoost'
}

def plot_roc_pr_curves():
    """Plot ROC and PR curves"""
    
    # Create 1x2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    
    # Store AUC values for legend
    roc_aucs = {}
    pr_aucs = {}
    
    # Plot curves for each algorithm
    for i, algorithm in enumerate(algorithms):
        color = colors[i]
        
        # Read prediction data
        pred_file = f'5_{algorithm}_predictions.csv'
        try:
            pred_data = pd.read_csv(pred_file)
            y_true = pred_data['y_true'].values
            y_prob = pred_data['y_prob'].values  
            
            # Calculate ROC curve data
            fpr, tpr, _ = roc_curve(y_true, y_prob)
            roc_auc = roc_auc_score(y_true, y_prob)
            roc_aucs[algorithm] = roc_auc
            
            # Plot ROC curve
            ax1.plot(fpr, tpr, color=color, linewidth=1.5, 
                    label=f'{algorithm} (AUC = {roc_auc:.4f})')
            
            # Calculate PR curve data
            precision, recall, _ = precision_recall_curve(y_true, y_prob)
            pr_auc = average_precision_score(y_true, y_prob)
            pr_aucs[algorithm] = pr_auc
            
            # Plot PR curve
            ax2.plot(recall, precision, color=color, linewidth=1.5,
                    label=f'{algorithm} (AUC = {pr_auc:.4f})')
            
        except FileNotFoundError:
            print(f"Warning: File {pred_file} not found")
            continue
        except Exception as e:
            print(f"Error processing file {pred_file}: {e}")
            continue
    
    # Set up ROC curve plot
    ax1.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.7, label='Random Classifier')
    ax1.set_xlim([0.0, 1.0])
    ax1.set_ylim([0.0, 1.05])
    ax1.set_xlabel('1-Specificity', fontsize=12)
    ax1.set_ylabel('Sensitivity', fontsize=12)
    # ax1.set_title('ROC Curves Comparison', fontsize=14)
    # ax1.legend(loc="lower right", frameon=True, fancybox=True, shadow=True)
    ax1.legend(loc="lower right")
    ax1.grid(True, alpha=0.3)
    ax1.text(-0.1, 1.05, '(a)', transform=ax1.transAxes, fontsize=14, fontweight='bold')
    
    # Set up PR curve plot
    # Calculate baseline (positive sample ratio)
    # If data is available, use the first algorithm's data to calculate baseline
    baseline_precision = 0.2822  # Default value, more accurate if calculated from data
    
    # Try to calculate the actual baseline from the first successfully read data
    for algorithm in algorithms:
        pred_file = f'5_{algorithm}_predictions.csv'
        try:
            pred_data = pd.read_csv(pred_file)
            y_true = pred_data['y_true'].values
            baseline_precision = np.mean(y_true)
            break
        except:
            continue
    
    ax2.axhline(y=baseline_precision, color='k', linestyle='--', linewidth=1, alpha=0.7, 
                label=f'Baseline (Precision = {baseline_precision:.4f})')
    
    ax2.set_xlim([0.0, 1.0])
    ax2.set_ylim([0.0, 1.05])
    ax2.set_xlabel('Recall', fontsize=12)
    ax2.set_ylabel('Precision', fontsize=12)
    # ax2.set_title('Precision-Recall Curves Comparison', fontsize=14)
    # ax2.legend(loc="upper right", frameon=True, fancybox=True, shadow=True)
    ax2.legend(loc="upper right")
    ax2.grid(True, alpha=0.3)
    ax2.text(-0.1, 1.05, '(b)', transform=ax2.transAxes, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    # Save plot
    plt.savefig('6_combined_roc_pr_curves.pdf', bbox_inches='tight')
    
    # Print AUC values summary
    print("\n=== ROC AUC Values ===")
    for algorithm in algorithms:
        if algorithm in roc_aucs:
            print(f"{algorithm}: {roc_aucs[algorithm]:.4f}")
    
    print("\n=== PR AUC Values ===")
    for algorithm in algorithms:
        if algorithm in pr_aucs:
            print(f"{algorithm}: {pr_aucs[algorithm]:.4f}")

if __name__ == "__main__":
    plot_roc_pr_curves()