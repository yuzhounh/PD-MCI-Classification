import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, precision_score, 
                           recall_score, f1_score, roc_auc_score, average_precision_score,
                           confusion_matrix, roc_curve, precision_recall_curve, cohen_kappa_score)
from sklearn.inspection import permutation_importance
import shap
import warnings
warnings.filterwarnings('ignore')

def load_data(train_file, test_file, feature_mapping_file):
    """Load training set, test set and feature mapping file"""
    train_data = pd.read_csv(train_file)
    test_data = pd.read_csv(test_file)
    feature_mapping = pd.read_csv(feature_mapping_file)
    
    # Create feature name mapping dictionary
    feature_name_mapping = dict(zip(feature_mapping['Feature Name'], feature_mapping['Abbreviation']))
    
    return train_data, test_data, feature_name_mapping

def prepare_data(data, site_col='SITE', label_col='MCI'):
    """Prepare data, separate features and labels"""
    # Remove subject ID column
    X = data.drop([site_col, label_col], axis=1)
    y = data[label_col]
    subjects = data[site_col]
    
    return X, y, subjects

def subject_level_cross_validation(X, y, subjects, n_folds=10, random_state=42):
    """
    Subject-level cross-validation - adopting "dimension reduction, splitting, dimension expansion" logic.
    """
    # Reduce to subject level
    subject_labels_series = pd.Series(y, index=subjects).groupby(level=0).max()
    unique_subjects = subject_labels_series.index.values
    subject_labels = subject_labels_series.values
    
    # Initialize stratified cross-validation
    sgkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    
    folds = []
    
    # Complete splitting at subject level
    for train_subject_idx, val_subject_idx in sgkf.split(unique_subjects, subject_labels, groups=unique_subjects):
        train_subjects_fold = unique_subjects[train_subject_idx]
        val_subjects_fold = unique_subjects[val_subject_idx]
        
        # Expand back to sample level
        train_sample_idx = np.where(np.isin(subjects, train_subjects_fold))[0]
        val_sample_idx = np.where(np.isin(subjects, val_subjects_fold))[0]
        
        folds.append((train_sample_idx, val_sample_idx))
    
    return folds

def f1_score_threshold(y_true, y_prob):
    """
    Calculate optimal threshold for maximum F1-score.
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    f1_scores = 2 * (precision * recall) / (precision + recall)
    f1_scores = np.nan_to_num(f1_scores)
    f1_scores = f1_scores[:-1]
    optimal_idx = np.argmax(f1_scores)
    return thresholds[optimal_idx]

def youden_threshold(y_true, y_prob):
    """Calculate optimal threshold based on Youden index"""
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    youden_scores = tpr - fpr  # Youden's J statistic
    optimal_idx = np.argmax(youden_scores)
    return thresholds[optimal_idx]

def calculate_all_metrics(y_true, y_pred, y_prob, threshold=None):
    """Calculate all evaluation metrics"""
    # Basic metrics
    accuracy = accuracy_score(y_true, y_pred)
    balanced_accuracy = balanced_accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    cohen_kappa = cohen_kappa_score(y_true, y_pred)
    auc_roc = roc_auc_score(y_true, y_prob)
    auc_pr = average_precision_score(y_true, y_prob)
    
    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    # Return results in specified order
    result = {
        'AUC-ROC': auc_roc,
        'AUC-PR': auc_pr,
        'Optimal Threshold': threshold if threshold is not None else 0.5,
        'Accuracy': accuracy,
        'Balanced Accuracy': balanced_accuracy,
        'Precision': precision,
        'Recall': recall,
        'Specificity': specificity,
        'F1-score': f1,
        "Cohen's Kappa": cohen_kappa
    }
    
    return result

def plot_roc_pr_curves(y_true, y_prob, model_name, save_prefix="5_"):
    """Plot ROC and PR curves"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # ROC curve
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc_roc = roc_auc_score(y_true, y_prob)
    
    ax1.plot(fpr, tpr, label=f'{model_name} (AUC = {auc_roc:.4f})', linewidth=2)
    ax1.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    ax1.set_xlabel('False Positive Rate')
    ax1.set_ylabel('True Positive Rate')
    ax1.set_title(f'{model_name} ROC Curve')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # PR curve
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    auc_pr = average_precision_score(y_true, y_prob)
    
    ax2.plot(recall, precision, label=f'{model_name} (AUC = {auc_pr:.4f})', linewidth=2)
    ax2.axhline(y=np.mean(y_true), color='k', linestyle='--', alpha=0.5, 
                label=f'Baseline ({np.mean(y_true):.3f})')
    ax2.set_xlabel('Recall')
    ax2.set_ylabel('Precision')
    ax2.set_title(f'{model_name} Precision-Recall Curve')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{save_prefix}{model_name}_roc_pr_curves.pdf', bbox_inches='tight')
    plt.close()

    # Save data
    predictions_data = pd.DataFrame({'y_true': y_true, 'y_prob': y_prob})
    predictions_data.to_csv(f'{save_prefix}{model_name}_predictions.csv', index=False)

def plot_feature_importance_comparison(importances_dict, feature_names, model_name, save_prefix="5_"):
    """Plot feature importance comparison"""
    n_methods = len(importances_dict)
    fig, axes = plt.subplots(1, n_methods, figsize=(6*n_methods, 8))
    
    if n_methods == 1:
        axes = [axes]
    
    for idx, (method_name, importance_values) in enumerate(importances_dict.items()):
        # Get sorting indices
        sorted_idx = np.argsort(np.abs(importance_values))
        sorted_features = [feature_names[i] for i in sorted_idx]
        sorted_values = importance_values[sorted_idx]
        
        # Plot horizontal bar chart
        bars = axes[idx].barh(range(len(sorted_values)), sorted_values)
        axes[idx].set_yticks(range(len(sorted_values)))
        axes[idx].set_yticklabels(sorted_features)
        axes[idx].set_xlabel('Importance')
        axes[idx].set_title(f'{model_name} - {method_name}')
        axes[idx].grid(True, alpha=0.3)
        
        # Use different colors for negative values
        for i, bar in enumerate(bars):
            if sorted_values[i] < 0:
                bar.set_color('#FF0051')  # 255, 0, 81
            else:
                bar.set_color('#008BFB')  # 0, 139, 251
    
    plt.tight_layout()
    plt.savefig(f'{save_prefix}{model_name}_feature_importance.pdf', bbox_inches='tight')
    plt.close()
    
    # Save data
    importance_df = pd.DataFrame(importances_dict, index=feature_names)
    importance_df.to_csv(f'{save_prefix}{model_name}_feature_importance.csv')

def plot_shap_summary(shap_values, X, feature_names, model_name, save_prefix="5_"):
    """Plot SHAP summary plots"""
    # Bar plot
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X, feature_names=feature_names, plot_type="bar", show=False)
    plt.title(f'{model_name} - SHAP Feature Importance (Bar)')
    plt.tight_layout()
    plt.savefig(f'{save_prefix}{model_name}_shap_bar.pdf', bbox_inches='tight')
    plt.close()
    
    # Summary plot
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X, feature_names=feature_names, show=False)
    plt.title(f'{model_name}', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{save_prefix}{model_name}_shap_summary.pdf', bbox_inches='tight')
    plt.savefig(f'{save_prefix}{model_name}_shap_summary.svg', bbox_inches='tight')
    plt.close()
    
    # Save SHAP values data
    shap_df = pd.DataFrame(shap_values, columns=feature_names)
    shap_df.to_csv(f'{save_prefix}{model_name}_shap_values.csv', index=False)

def get_permutation_importance(model, X, y, feature_names, random_state=42):
    """Calculate permutation importance"""
    perm_importance = permutation_importance(model, X, y, 
                                           n_repeats=10, 
                                           random_state=random_state, 
                                           scoring='roc_auc')
    return perm_importance.importances_mean

def print_results(model_name, metrics, optimal_threshold):
    """Print results"""
    print(f"\n{'='*50}")
    print(f"{model_name} Model Results")
    print(f"{'='*50}")
    print(f"Optimal Threshold: {optimal_threshold:.4f}")
    
    print(f"\nConfusion Matrix:")
    print(metrics['Confusion Matrix'])

    print("\nEvaluation Metrics:")
    for metric, value in metrics.items():
        if metric != 'Confusion Matrix':
            print(f"{metric}: {value}")
