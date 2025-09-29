import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import average_precision_score
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# Set font to Arial for plots
plt.rcParams['font.sans-serif'] = ['Arial']

# ====================== Core Modular Functions ======================

def create_subject_proxy_labels(y, subjects):
    """
    Create subject-level proxy labels (if any sample is 1, subject is 1 rule)
    
    Parameters:
    -----------
    y : array-like
        Sample-level labels
    subjects : array-like  
        Subject IDs corresponding to samples
        
    Returns:
    --------
    y_proxy : array
        Sample-level proxy labels (each sample uses its subject's proxy label)
    subject_labels_map : dict
        Subject to proxy label mapping
    """
    # Ensure inputs are numpy arrays
    y = np.asarray(y)
    subjects = np.asarray(subjects)
    
    # Create subject label mapping: each subject uses "if any is 1, then 1" rule
    subject_labels_map = pd.Series(y, index=subjects).groupby(level=0).max()
    
    # Assign each sample its subject's proxy label
    y_proxy = pd.Series(subjects).map(subject_labels_map).values
    
    return y_proxy, subject_labels_map.to_dict()


def subject_level_cross_validation(X, y, subjects, n_folds=10, random_state=42):
    """
    Subject-level cross validation - adopts "reduce, split, expand" logic.
    
    Parameters:
    -----------
    X : array-like, shape (n_samples, n_features)
        Feature matrix
    y : array-like, shape (n_samples,)
        Label array
    subjects : array-like, shape (n_samples,)
        Subject ID array
    n_folds : int, default=10
        Number of cross-validation folds
    random_state : int, default=42
        Random seed
        
    Returns:
    --------
    folds : list of tuples
        Each tuple contains sample-level (train_idx, val_idx)
    """
    # Reduce to subject level
    subject_labels_series = pd.Series(y, index=subjects).groupby(level=0).max()
    unique_subjects = subject_labels_series.index.values
    subject_labels = subject_labels_series.values
    
    # Initialize stratified cross-validator
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

def print_split_statistics(train_data, test_data, site_col='SITE', label_col='MCI'):
    """
    Modular function to print data split statistics
    """
    print(f"Training set shape: {train_data.shape}")
    print(f"Test set shape: {test_data.shape}")
    
    # Sample-level statistics
    print(f"\nSample-level label distribution:")
    print(f"Training set label distribution: {train_data[label_col].value_counts().to_dict()}")
    print(f"Test set label distribution: {test_data[label_col].value_counts().to_dict()}")

    train_class_1_ratio = train_data[label_col].value_counts(normalize=True).get(1, 0)
    test_class_1_ratio = test_data[label_col].value_counts(normalize=True).get(1, 0)
    difference = train_class_1_ratio - test_class_1_ratio
    
    print(f"Proportion of class 1 in training set: {train_class_1_ratio:.4f}")
    print(f"Proportion of class 1 in test set: {test_class_1_ratio:.4f}")
    print(f"Difference between proportions: {difference:.4f}")
    
    # Subject-level statistics
    train_subjects = train_data[site_col].unique()
    test_subjects = test_data[site_col].unique()
    
    print(f"Number of subjects in training set: {len(train_subjects)}")
    print(f"Number of subjects in test set: {len(test_subjects)}")
    
    # Check for overlap
    overlapping_subjects = set(train_subjects) & set(test_subjects)
    if overlapping_subjects:
        print(f"Warning: Subject overlap between training and test sets: {overlapping_subjects}")
    else:
        print("No subject overlap between training and test sets.")
    
    # Subject-level label distribution
    _, train_subject_labels_map = create_subject_proxy_labels(
        train_data[label_col].values, train_data[site_col].values
    )
    _, test_subject_labels_map = create_subject_proxy_labels(
        test_data[label_col].values, test_data[site_col].values
    )
    
    train_subject_labels = pd.Series(list(train_subject_labels_map.values()))
    test_subject_labels = pd.Series(list(test_subject_labels_map.values()))
    
    print(f"\nSubject-level label distribution (if any is 1, then 1 rule):")
    print(f"Training set subject label distribution: {train_subject_labels.value_counts().to_dict()}")
    print(f"Test set subject label distribution: {test_subject_labels.value_counts().to_dict()}")
    
    train_subject_class_1_ratio = train_subject_labels.value_counts(normalize=True).get(1, 0)
    test_subject_class_1_ratio = test_subject_labels.value_counts(normalize=True).get(1, 0)
    subject_difference = train_subject_class_1_ratio - test_subject_class_1_ratio
    
    print(f"Proportion of class 1 subjects in training set: {train_subject_class_1_ratio:.4f}")
    print(f"Proportion of class 1 subjects in test set: {test_subject_class_1_ratio:.4f}")
    print(f"Subject-level proportion difference: {subject_difference:.4f}")

# ====================== Refactored Main Functions ======================

def stratified_group_split(data, test_size=0.3, random_state=42, site_col='SITE', label_col='MCI'):
    """
    Split data using subject-level stratified sampling (refactored version using proxy label mechanism)
    Ensures data from the same subject is either in training or test set
    """
    # Create proxy labels
    y_proxy, subject_labels_map = create_subject_proxy_labels(
        data[label_col].values, data[site_col].values
    )
    
    # Get unique subjects and corresponding proxy labels
    unique_subjects = np.array(list(subject_labels_map.keys()))
    subject_proxy_labels = np.array(list(subject_labels_map.values()))
    
    # Perform stratified split at subject level
    train_subjects, test_subjects = train_test_split(
        unique_subjects, 
        test_size=test_size, 
        stratify=subject_proxy_labels, 
        random_state=random_state
    )
    
    # Split data based on subjects
    train_data = data[data[site_col].isin(train_subjects)]
    test_data = data[data[site_col].isin(test_subjects)]
    
    # Print statistics
    print_split_statistics(train_data, test_data, site_col, label_col)
    
    return train_data, test_data

def evaluate_model_cv(X, y, subjects, model, cv_folds, scoring_func=average_precision_score):
    """
    Modular function for model evaluation using cross-validation
    
    Parameters:
    -----------
    X : array-like
        Feature matrix
    y : array-like
        Label array
    subjects : array-like
        Subject IDs
    model : sklearn estimator
        Model to evaluate
    cv_folds : list of tuples
        Cross-validation fold indices
    scoring_func : callable
        Scoring function
        
    Returns:
    --------
    scores : array
        Score for each fold
    """
    scores = []
    
    for fold_idx, (train_idx, val_idx) in enumerate(cv_folds):
        X_train_fold = X[train_idx]
        y_train_fold = y[train_idx]
        X_val_fold = X[val_idx]
        y_val_fold = y[val_idx]
        
        # Validate split correctness
        train_subjects_in_fold = set(subjects[train_idx])
        val_subjects_in_fold = set(subjects[val_idx])
        if train_subjects_in_fold & val_subjects_in_fold:
            print(f"Warning: Subject overlap in fold {fold_idx+1} between training and validation sets")
        
        # Train model
        model.fit(X_train_fold, y_train_fold)
        
        # Predict and score
        if hasattr(model, 'predict_proba'):
            y_pred = model.predict_proba(X_val_fold)[:, 1]
        else:
            y_pred = model.predict(X_val_fold)
            
        score = scoring_func(y_val_fold, y_pred)
        scores.append(score)
    
    return np.array(scores)

def compute_coefficient_path(X, y, lambda_seq, random_state=42):
    """
    Modular function to compute LASSO coefficient path
    
    Parameters:
    -----------
    X : array-like
        Feature matrix
    y : array-like
        Label array
    lambda_seq : array-like
        Regularization parameter sequence
    random_state : int
        Random seed
        
    Returns:
    --------
    coefficients_path : array, shape (n_lambdas, n_features)
        Coefficient path
    """
    coefficients_path = np.zeros((len(lambda_seq), X.shape[1]))
    
    for i, lam in enumerate(lambda_seq):
        model = LogisticRegression(
            penalty='l1',
            solver='liblinear',
            C=1.0/lam,
            class_weight='balanced',
            random_state=random_state,
            max_iter=1000
        )
        model.fit(X, y)
        coefficients_path[i] = model.coef_[0]
    
    # Save coefficient path as CSV file
    coefficients_df = pd.DataFrame(coefficients_path)
    coefficients_df.index = lambda_seq
    coefficients_df.to_csv('4_lasso_coefficients_path.csv', index_label='lambda')
    
    return coefficients_path

def lasso_kfold_cv_with_coefficients(X, y, subjects, lambda_seq, n_folds=10, random_state=42):
    """
    LASSO hyperparameter tuning using K-fold cross-validation (refactored version)
    Uses proxy label mechanism and modular design
    """
    # Generate cross-validation folds
    cv_folds = subject_level_cross_validation(X, y, subjects, n_folds, random_state)
    
    # Create proxy labels for statistical output
    y_proxy, subject_labels_map = create_subject_proxy_labels(y, subjects)
    unique_subjects = np.array(list(subject_labels_map.keys()))
    subject_proxy_labels = np.array(list(subject_labels_map.values()))
    
    # Print basic information
    print(f"Starting {n_folds}-fold cross-validation...")
    print(f"Total number of subjects: {len(unique_subjects)}")
    print(f"Subject label distribution (if any is 1, then 1 rule): {np.unique(subject_proxy_labels, return_counts=True)}")
    
    # Store AUC-PR scores for each lambda
    auc_pr_scores = np.zeros((n_folds, len(lambda_seq)))
    
    # Cross-validation for each lambda value
    for i, lam in enumerate(lambda_seq):
        model = LogisticRegression(
            penalty='l1', 
            solver='liblinear', 
            C=1.0/lam, 
            class_weight='balanced',
            random_state=random_state,
            max_iter=1000
        )
        
        # Use modular evaluation function
        scores = evaluate_model_cv(X, y, subjects, model, cv_folds, average_precision_score)
        auc_pr_scores[:, i] = scores
        
        if (i + 1) % 5 == 0:  # Print progress every 5 lambdas
            print(f"Lambda progress: {(i + 1) / len(lambda_seq) * 100:.1f}%")
    
    # Calculate mean AUC-PR and standard deviation
    mean_auc_pr = np.mean(auc_pr_scores, axis=0)
    std_auc_pr = np.std(auc_pr_scores, axis=0)
    
    print(f"Cross-validation completed with {n_folds} folds")
    print(f"Mean AUC-PR: {np.mean(mean_auc_pr):.4f} ± {np.mean(std_auc_pr):.4f}")
    print(f"Best AUC-PR: {np.max(mean_auc_pr):.4f} (λ={lambda_seq[np.argmax(mean_auc_pr)]:.6f})")
    
    # Use modular function to compute coefficient path
    print("\nComputing coefficient path on full training set...")
    coefficients_path = compute_coefficient_path(X, y, lambda_seq, random_state)
    
    # Get optimal lambda index
    best_lambda_idx = np.argmax(mean_auc_pr)

    return mean_auc_pr, std_auc_pr, coefficients_path, best_lambda_idx

# ====================== Other Helper Functions Remain Unchanged ======================

def standardize_data(train_data, test_data):
    """Data standardization"""
    # Use column names to specify subject and label columns
    site_col = 'SITE'
    label_col = 'MCI'
    
    # Separate features and labels
    feature_cols = [col for col in train_data.columns if col not in [site_col, label_col]]
    X_train = train_data[feature_cols]
    y_train = train_data[label_col]
    X_test = test_data[feature_cols]
    y_test = test_data[label_col]
    
    # Standardize
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Convert to DataFrame
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)
    
    # Recombine data
    train_scaled = pd.concat([train_data[[site_col]], X_train_scaled, train_data[[label_col]]], axis=1)
    test_scaled = pd.concat([test_data[[site_col]], X_test_scaled, test_data[[label_col]]], axis=1)
    
    return train_scaled, test_scaled, scaler

def plot_results(lambda_seq, mean_auc_pr_scores, std_auc_pr_scores, coefficients_path, feature_names, feature_dict, best_lambda_idx):
    """Plot AUC-PR and coefficient path"""
    
    # Map feature names
    abbreviated_names = [feature_dict.get(name, name) for name in feature_names]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    
    # Subplot (a): AUC-PR vs Lambda
    ax1.plot(lambda_seq, mean_auc_pr_scores, '-', linewidth=1.5, marker='o', markersize=4)
    ax1.fill_between(lambda_seq, mean_auc_pr_scores - std_auc_pr_scores, mean_auc_pr_scores + std_auc_pr_scores, alpha=0.2)
    ax1.set_xlabel('$\\lambda$')
    ax1.set_ylabel('AUC-PR')
    # ax1.set_title('AUC-PR vs Lambda (10-Fold CV)')
    ax1.set_xscale('log')
    ax1.grid(True, alpha=0.3)
    ax1.axvline(lambda_seq[best_lambda_idx], color='r', linestyle='--', alpha=0.7, label=f'Best $\\lambda$={lambda_seq[best_lambda_idx]:.4f}')
    ax1.plot(lambda_seq[best_lambda_idx], mean_auc_pr_scores[best_lambda_idx], 'ro', markersize=8)
    ax1.legend(loc='lower left')
    ax1.text(-0.1, 1.05, '(a)', transform=ax1.transAxes, fontsize=14, fontweight='bold')
    
    # Subplot (b): Coefficient path
    feature_indices_to_plot = []
    for i in range(len(feature_names)):
        if np.any(np.abs(coefficients_path[:, i]) > 1e-6):
            feature_indices_to_plot.append(i)
    
    # If too many features, only show the most important ones
    if len(feature_indices_to_plot) > 20:
        best_coeffs = np.abs(coefficients_path[best_lambda_idx])
        top_features = np.argsort(best_coeffs)[-20:]
        feature_indices_to_plot = [i for i in feature_indices_to_plot if i in top_features]
    
    for i in feature_indices_to_plot:
        ax2.plot(lambda_seq, coefficients_path[:, i], label=abbreviated_names[i], linewidth=1.5)
    
    ax2.set_xlabel('$\\lambda$')
    ax2.set_ylabel('Coefficient')
    # ax2.set_title('Coefficient Path')
    ax2.set_xscale('log')
    ax2.grid(True, alpha=0.3)
    ax2.axvline(lambda_seq[best_lambda_idx], color='r', linestyle='--', alpha=0.7)
    ax2.text(-0.1, 1.05, '(b)', transform=ax2.transAxes, fontsize=14, fontweight='bold')
    
    # Add legend
    if len(feature_indices_to_plot) <= 15:
        ax2.legend(bbox_to_anchor=(1.03, 1.02), loc='upper left')
    else:
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
    
    plt.tight_layout()
    
    # Save as PDF
    with PdfPages('4_lasso_plot.pdf') as pdf:
        pdf.savefig(fig, bbox_inches='tight')
    # plt.show()
    
    return best_lambda_idx

def extract_selected_features(selected_features):
    """
    Extract selected features and save as new CSV files
    
    Parameters:
    -----------
    selected_features : list
        List of selected feature column names
    """
    print(f"\n=== Step 8: Extract Selected Features and Save New Dataset ===")
    
    # Fixed file names
    train_file = 'PPMI_6_train.csv'
    test_file = 'PPMI_6_test.csv'
    output_train = 'PPMI_7_train.csv'
    output_test = 'PPMI_7_test.csv'
    site_col = 'SITE'
    label_col = 'MCI'
    
    # Read original data
    train_data = pd.read_csv(train_file)
    test_data = pd.read_csv(test_file)
    
    print(f"Original training set shape: {train_data.shape}")
    print(f"Original test set shape: {test_data.shape}")
    
    # Determine columns to keep: SITE + selected_features + MCI
    columns_to_keep = [site_col] + selected_features + [label_col]
    
    # Check if all columns exist
    missing_train = [col for col in columns_to_keep if col not in train_data.columns]
    missing_test = [col for col in columns_to_keep if col not in test_data.columns]
    
    if missing_train:
        print(f"Warning: Missing columns in training set: {missing_train}")
    if missing_test:
        print(f"Warning: Missing columns in test set: {missing_test}")
    
    # Extract specified columns
    train_selected = train_data[columns_to_keep]
    test_selected = test_data[columns_to_keep]
    
    # Save new files
    train_selected.to_csv(output_train, index=False)
    test_selected.to_csv(output_test, index=False)
    
    print(f"Training set shape after extraction: {train_selected.shape}")
    print(f"Test set shape after extraction: {test_selected.shape}")
    print(f"Columns kept: {columns_to_keep}")
    print(f"Files saved: {output_train}, {output_test}")
    
    return train_selected, test_selected

def main():
    """Main function"""
    # 1. Load data
    print("=== Step 1: Load Data ===")
    data = pd.read_csv('PPMI_5_remove_PATNO.csv')
    print(f"Original data shape: {data.shape}")

    feature_mapping = pd.read_csv('../PPMI_feature_mapping.csv')
    feature_dict = dict(zip(feature_mapping['Feature Name'], feature_mapping['Abbreviation']))
    print(f"Loaded {len(feature_dict)} feature mappings")
    
    # 2. Data splitting
    print("\n=== Step 2: Data Splitting (Using Proxy Label Mechanism) ===")
    train_data, test_data = stratified_group_split(data)
    
    # 3. Data standardization
    print("\n=== Step 3: Data Standardization ===")
    train_scaled, test_scaled, scaler = standardize_data(train_data, test_data)
    
    # Save standardized data
    train_scaled.to_csv('PPMI_6_train.csv', index=False)
    test_scaled.to_csv('PPMI_6_test.csv', index=False)
    print("Saved standardized data: PPMI_6_train.csv, PPMI_6_test.csv")
    
    # 4. K-fold cross-validation hyperparameter tuning and coefficient path computation
    print("\n=== Step 4: K-fold Cross-validation Hyperparameter Tuning and Coefficient Path Computation (Modular Refactored Version) ===")
    lambda_seq = 10**np.arange(0, 3, 0.05)
    
    # Get features using column names
    site_col = 'SITE'
    label_col = 'MCI'
    feature_cols = [col for col in train_scaled.columns if col not in [site_col, label_col]]
    
    X_train_scaled = train_scaled[feature_cols]
    y_train = train_scaled[label_col]
    subjects_train = train_scaled[site_col].values

    mean_auc_pr, std_auc_pr, coefficients_path, best_lambda_idx = lasso_kfold_cv_with_coefficients(
        X_train_scaled.values, y_train.values, subjects_train, lambda_seq, n_folds=10
    )
    
    # 5. Plotting and result analysis
    print("\n=== Step 5: Result Analysis and Visualization ===")
    feature_names = X_train_scaled.columns.tolist()
    plot_results(lambda_seq, mean_auc_pr, std_auc_pr, coefficients_path, feature_names, feature_dict, best_lambda_idx)
    
    # 6. Output optimal results
    best_lambda = lambda_seq[best_lambda_idx]
    best_auc_pr = mean_auc_pr[best_lambda_idx]
    best_coeffs = coefficients_path[best_lambda_idx]
    
    print(f"\nOptimal lambda: {best_lambda:.6f}")
    print(f"Optimal AUC-PR: {best_auc_pr:.4f} ± {std_auc_pr[best_lambda_idx]:.4f}")
    
    # Select non-zero features
    non_zero_mask = np.abs(best_coeffs) > 1e-6
    selected_features = [feature_names[i] for i in range(len(feature_names)) if non_zero_mask[i]]
    selected_weights = [best_coeffs[i] for i in range(len(best_coeffs)) if non_zero_mask[i]]
    
    print(f"\nNumber of selected features: {len(selected_features)}")
    print(f"Feature selection ratio: {len(selected_features)}/{len(feature_names)} ({len(selected_features)/len(feature_names)*100:.1f}%)")
    
    # 7. Save feature weight results
    results_df = pd.DataFrame({
        'Feature': selected_features,
        'Abbreviation': [feature_dict.get(f, f) for f in selected_features],
        'Weight': selected_weights
    })
    results_df = results_df.sort_values('Weight', key=abs, ascending=False)
    results_df.to_csv('4_feature_weight.csv', index=False)
    
    print("\nSelected features and their weights:")
    print(results_df.to_string(index=False))
    print(f"\nWeight statistics:")
    print(f"Maximum weight: {results_df['Weight'].abs().max():.4f}")
    print(f"Mean absolute weight: {results_df['Weight'].abs().mean():.4f}")
    print(f"Number of positive weight features: {(results_df['Weight'] > 0).sum()}")
    print(f"Number of negative weight features: {(results_df['Weight'] < 0).sum()}")
    
    # 8. Plot feature weight bar chart
    plt.figure(figsize=(6, 4))
    
    # Sort by absolute weight (descending)
    results_sorted = results_df.copy().sort_values('Weight', key=abs, ascending=True)
    
    # Set colors based on weight values
    colors = ['#2E86AB' if w > 0 else '#F24236' for w in results_sorted['Weight']]
    
    # Create horizontal bar chart, all weight values plotted in positive direction (using absolute values)
    bars = plt.barh(range(len(results_sorted)), results_sorted['Weight'].abs(), color=colors, alpha=0.8)
    
    # Set y-axis labels
    plt.yticks(range(len(results_sorted)), results_sorted['Abbreviation'])
    plt.xlabel('Weight')
    # plt.ylabel('Features', fontsize=14, fontweight='bold')
    # plt.title('LASSO Feature Weights for PD-MCI Classification', fontsize=16, fontweight='bold', pad=20)

    # Set x-axis range
    plt.xlim(0, 0.60)

    # Add weight value labels
    for i, (bar, weight) in enumerate(zip(bars, results_sorted['Weight'])):
        plt.text(abs(weight) + 0.005, 
                i, f'{weight:.4f}', 
                ha='left', 
                va='center')
    
    # Set legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='#2E86AB', alpha=0.8, label='Positive'),
                      Patch(facecolor='#F24236', alpha=0.8, label='Negative')]
    plt.legend(handles=legend_elements, loc='lower right')
    
    # Adjust layout and style
    plt.grid(axis='x', alpha=0.3, linestyle='--')
    plt.tight_layout()
    
    # Save figure
    plt.savefig('4_feature_weight.pdf', dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    print("- 4_feature_weight.pdf: Feature weight bar chart")
    
    # 9. Extract selected features and save new dataset
    extract_selected_features(selected_features)
    
    print("\n=== Analysis Complete ===")
    print("Output files:")
    print("- PPMI_6_train.csv: Standardized training set")
    print("- PPMI_6_test.csv: Standardized test set")
    print("- PPMI_7_train.csv: Training set with selected features only")
    print("- PPMI_7_test.csv: Test set with selected features only")
    print("- 4_feature_weight.csv: Feature weight results (AUC-PR optimized)")
    print("- 4_lasso_plot.pdf: LASSO analysis plot (AUC-PR optimized)")
    print("- 4_feature_weight.pdf: Feature weight bar chart")

if __name__ == "__main__":
    main()