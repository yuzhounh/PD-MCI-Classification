import pandas as pd
import numpy as np
import scipy.stats as st
from sklearn.metrics import roc_auc_score

def delong_test(y_true, y_pred1, y_pred2, alpha=0.05):
    """
    Improved DeLong's test for comparing two correlated ROC curves.
    
    Parameters:
    y_true: True binary labels (0 or 1)
    y_pred1: Prediction probabilities from model 1
    y_pred2: Prediction probabilities from model 2
    alpha: Significance level (default: 0.05)
    
    Returns:
    dict: Dictionary containing detailed statistical information
    """
    # Input validation and conversion
    y_true = np.asarray(y_true).astype(int)
    y_pred1 = np.asarray(y_pred1).astype(float)
    y_pred2 = np.asarray(y_pred2).astype(float)
    
    # Check input data validity
    if len(y_true) != len(y_pred1) or len(y_true) != len(y_pred2):
        raise ValueError("Input arrays have inconsistent lengths")
    
    if len(np.unique(y_true)) != 2:
        raise ValueError("y_true must be binary classification labels")
        
    if not set(np.unique(y_true)).issubset({0, 1}):
        raise ValueError("y_true must contain 0 and 1")
    
    if np.any((y_pred1 < 0) | (y_pred1 > 1)) or np.any((y_pred2 < 0) | (y_pred2 > 1)):
        raise ValueError("Prediction probabilities must be in [0,1] range")
    
    # Separate positive and negative class prediction probabilities
    pos_mask = (y_true == 1)
    neg_mask = (y_true == 0)
    
    y_pred1_pos = y_pred1[pos_mask]
    y_pred1_neg = y_pred1[neg_mask]
    y_pred2_pos = y_pred2[pos_mask]
    y_pred2_neg = y_pred2[neg_mask]

    m = len(y_pred1_pos)  # Number of positive samples
    n = len(y_pred1_neg)  # Number of negative samples

    if m == 0 or n == 0:
        raise ValueError("Data must contain both positive and negative samples")
    
    if m < 2 or n < 2:
        raise ValueError("Both positive and negative sample counts must be ≥2 to calculate covariance")

    # --- Calculate structural components V ---
    # Use more stable numerical calculation method
    v10_1 = np.zeros(m)
    v10_2 = np.zeros(m)
    for i in range(m):
        v10_1[i] = np.mean(y_pred1_pos[i] > y_pred1_neg)
        v10_2[i] = np.mean(y_pred2_pos[i] > y_pred2_neg)
    
    v01_1 = np.zeros(n)
    v01_2 = np.zeros(n)
    for j in range(n):
        v01_1[j] = np.mean(y_pred1_neg[j] < y_pred1_pos)
        v01_2[j] = np.mean(y_pred2_neg[j] < y_pred2_pos)

    # --- Calculate covariance matrix ---
    # Use unbiased estimator (ddof=1)
    try:
        s10 = np.cov(v10_1, v10_2, ddof=1)
        s01 = np.cov(v01_1, v01_2, ddof=1)
    except np.linalg.LinAlgError:
        # Handle covariance matrix calculation failure
        return {
            'p_value': np.nan,
            'z_score': np.nan,
            'auc1': roc_auc_score(y_true, y_pred1),
            'auc2': roc_auc_score(y_true, y_pred2),
            'auc_diff': np.nan,
            'var_diff': np.nan,
            'error': 'Covariance matrix calculation failed'
        }

    # --- Calculate statistics ---
    auc1 = roc_auc_score(y_true, y_pred1)
    auc2 = roc_auc_score(y_true, y_pred2)
    auc_diff = auc1 - auc2

    # Calculate variance of AUC difference
    var_auc1 = s10[0, 0] / m + s01[0, 0] / n
    var_auc2 = s10[1, 1] / m + s01[1, 1] / n
    cov_auc1_auc2 = s10[0, 1] / m + s01[0, 1] / n
    
    var_diff = var_auc1 + var_auc2 - 2 * cov_auc1_auc2

    # Numerical stability check
    if var_diff <= 1e-12:
        # Variance close to 0, models are almost identical
        z_score = 0.0
        p_value = 1.0
    else:
        z_score = auc_diff / np.sqrt(var_diff)
        # Two-sided test
        p_value = 2.0 * (1 - st.norm.cdf(np.abs(z_score)))

    # Calculate confidence interval
    margin_error = st.norm.ppf(1 - alpha/2) * np.sqrt(var_diff)
    ci_lower = auc_diff - margin_error
    ci_upper = auc_diff + margin_error

    return {
        'p_value': p_value,
        'z_score': z_score,
        'auc1': auc1,
        'auc2': auc2,
        'auc_diff': auc_diff,
        'var_diff': var_diff,
        'confidence_interval': (ci_lower, ci_upper),
        'significant': p_value < alpha,
        'sample_sizes': {'positive': m, 'negative': n}
    }


def run_single_delong_analysis(file_common_test_set, file_model_A_preds, file_model_B_preds):
    """
    Run single DeLong analysis and return result dictionary
    """
    try:
        # Data loading
        test_df = pd.read_csv(file_common_test_set)
        true_labels = test_df['MCI'].values.astype(int)
        
        preds_A_df = pd.read_csv(file_model_A_preds)
        probs_A = preds_A_df['y_prob'].values
        
        preds_B_df = pd.read_csv(file_model_B_preds)
        probs_B = preds_B_df['y_prob'].values
        
        # Execute DeLong test
        result = delong_test(true_labels, probs_A, probs_B)
        
        return result
        
    except Exception as e:
        return {
            'error': str(e),
            'p_value': np.nan,
            'z_score': np.nan,
            'auc1': np.nan,
            'auc2': np.nan,
            'auc_diff': np.nan,
            'var_diff': np.nan,
            'confidence_interval': (np.nan, np.nan),
            'significant': False,
            'sample_sizes': {'positive': np.nan, 'negative': np.nan}
        }


def create_results_table(algorithms, file_common_test_set, model_A_dir, model_B_dir):
    """
    Create DeLong test results table
    
    Parameters:
    algorithms: List of algorithms
    file_common_test_set: Common test set file path
    model_A_dir: Model A prediction file directory
    model_B_dir: Model B prediction file directory
    
    Returns:
    pandas.DataFrame: Results table
    """
    results = []
    
    print("Executing DeLong test analysis...")
    print("="*80)
    
    for algorithm in algorithms:
        print(f"Analyzing {algorithm} algorithm...")
        
        file_model_A_preds = f"{model_A_dir}/5_{algorithm}_predictions.csv"
        file_model_B_preds = f"{model_B_dir}/5_{algorithm}_predictions.csv"
        
        result = run_single_delong_analysis(file_common_test_set, file_model_A_preds, file_model_B_preds)
        
        # Build table row data
        if 'error' in result and pd.isna(result['auc1']):
            # Handle error cases
            row = {
                'Algorithm': algorithm,
                'Model_A_AUC': 'Error',
                'Model_B_AUC': 'Error',
                'AUC_Difference': 'Error',
                'Z_Statistic': 'Error',
                'P_Value': 'Error',
                '95%CI_Lower': 'Error',
                '95%CI_Upper': 'Error',
                'Significant': 'Error',
                'Conclusion': f"Error: {result['error']}",
                'Positive_Samples': 'Error',
                'Negative_Samples': 'Error'
            }
        else:
            # Normal cases
            ci_lower, ci_upper = result['confidence_interval']
            
            # Determine conclusion
            if result['significant']:
                if result['auc_diff'] > 0:
                    conclusion = "Model A significantly better than Model B"
                else:
                    conclusion = "Model B significantly better than Model A"
            else:
                conclusion = "No significant difference between models"
            
            row = {
                'Algorithm': algorithm,
                'Model_A_AUC': f"{result['auc1']:.4f}",
                'Model_B_AUC': f"{result['auc2']:.4f}",
                'AUC_Difference': f"{result['auc_diff']:.4f}",
                'Z_Statistic': f"{result['z_score']:.4f}",
                'P_Value': f"{result['p_value']:.6f}",
                '95%CI_Lower': f"{ci_lower:.4f}",
                '95%CI_Upper': f"{ci_upper:.4f}",
                'Significant': "Yes" if result['significant'] else "No",
                'Conclusion': conclusion,
                'Positive_Samples': result['sample_sizes']['positive'],
                'Negative_Samples': result['sample_sizes']['negative']
            }
        
        results.append(row)
        print(f"{algorithm} analysis completed")
    
    # Create DataFrame
    df = pd.DataFrame(results)
    
    return df


def display_results_table(df, save_path=None):
    """
    Display and save results table
    
    Parameters:
    df: Results DataFrame
    save_path: Save path (optional)
    """
    print("\n" + "="*80)
    print("DeLong Test Results Summary Table")
    print("="*80)
    
    # Set pandas display options
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', 20)
    
    print(df.to_string(index=False))
    
    # Save results
    if save_path:
        df.to_csv(save_path, index=False, encoding='utf-8-sig')
        print(f"\nResults saved to: {save_path}")
    
    # Summary statistics
    print(f"\nSummary Statistics:")
    if 'Error' not in df['Significant'].values:
        significant_count = (df['Significant'] == 'Yes').sum()
        total_count = len(df)
        print(f"- Total algorithms: {total_count}")
        print(f"- Algorithms with significant difference: {significant_count}")
        print(f"- Algorithms with no significant difference: {total_count - significant_count}")
        
        # Count which model is better
        model_a_better = df[df['Conclusion'].str.contains('Model A significantly better', na=False)].shape[0]
        model_b_better = df[df['Conclusion'].str.contains('Model B significantly better', na=False)].shape[0]
        
        if model_a_better > 0:
            print(f"- Algorithms where Model A significantly better than Model B: {model_a_better}")
        if model_b_better > 0:
            print(f"- Algorithms where Model B significantly better than Model A: {model_b_better}")
    
    return df


# Main function
if __name__ == "__main__":
    # Define algorithm list
    algorithms = ['LR', 'SVM', 'RF', 'XGBoost']
    
    # Define file paths
    file_common_test_set = "../PPMI_6_test.csv"
    model_A_dir = "../"
    model_B_dir = "../supplementary_experiment_2/"
    
    # Create results table
    results_df = create_results_table(algorithms, file_common_test_set, model_A_dir, model_B_dir)
    
    # Display and save results
    results_df = display_results_table(results_df, save_path="7_delong_test.csv")
    
    print("\nDeLong test analysis completed!")