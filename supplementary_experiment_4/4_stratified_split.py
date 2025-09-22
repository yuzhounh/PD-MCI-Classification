import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

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

def print_split_statistics(train_data, test_data, subject_col='PATNO', label_col='MCI'):
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
    train_subjects = train_data[subject_col].unique()
    test_subjects = test_data[subject_col].unique()
    
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
        train_data[label_col].values, train_data[subject_col].values
    )
    _, test_subject_labels_map = create_subject_proxy_labels(
        test_data[label_col].values, test_data[subject_col].values
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

def stratified_group_split(data, test_size=0.3, random_state=42, subject_col='PATNO', label_col='MCI'):
    """
    Split data using subject-level stratified sampling (refactored version using proxy label mechanism)
    Ensures data from the same subject is either in training or test set
    """
    # Create proxy labels
    y_proxy, subject_labels_map = create_subject_proxy_labels(
        data[label_col].values, data[subject_col].values
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
    train_data = data[data[subject_col].isin(train_subjects)]
    test_data = data[data[subject_col].isin(test_subjects)]
    
    # Print statistics
    print_split_statistics(train_data, test_data, subject_col, label_col)
    
    return train_data, test_data

# ====================== Other Helper Functions Remain Unchanged ======================

def standardize_data(train_data, test_data):
    """Data standardization"""
    # Use column names to specify subject and label columns
    subject_col = 'PATNO'
    label_col = 'MCI'
    
    # Separate features and labels
    feature_cols = [col for col in train_data.columns if col not in [subject_col, label_col]]
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
    train_scaled = pd.concat([train_data[[subject_col]], X_train_scaled, train_data[[label_col]]], axis=1)
    test_scaled = pd.concat([test_data[[subject_col]], X_test_scaled, test_data[[label_col]]], axis=1)
    
    return train_scaled, test_scaled, scaler

def main():
    """Main function"""
    # 1. Load data
    print("=== Step 1: Load Data ===")
    data = pd.read_csv('PPMI_5_label.csv')
    print(f"Original data shape: {data.shape}")

    feature_mapping = pd.read_csv('PPMI_feature_mapping.csv')
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
    
    # print("Output files:")
    print("- PPMI_6_train.csv: Standardized training set")
    print("- PPMI_6_test.csv: Standardized test set")

if __name__ == "__main__":
    main()

