import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import chi2_contingency, mannwhitneyu, ttest_ind, fisher_exact
from statsmodels.stats.multitest import fdrcorrection
import warnings
warnings.filterwarnings('ignore')

def determine_variable_type(series):
    """Determine variable type"""
    unique_values = series.dropna().unique()
    
    # If only two unique values, consider it as binary variable
    if len(unique_values) == 2:
        return 'Binary'
    # If unique values < 10 and all are integers, consider it as categorical variable
    elif len(unique_values) < 10 and all(isinstance(x, (int, np.integer)) or (isinstance(x, float) and x.is_integer()) for x in unique_values):
        return 'Categorical'
    else:
        return 'Continuous'

def perform_statistical_test(data, group_col, feature_col, var_type):
    """Perform statistical test"""
    group_0 = data[data[group_col] == 0][feature_col].dropna()
    group_1 = data[data[group_col] == 1][feature_col].dropna()
    
    if var_type == 'Binary' or var_type == 'Categorical':
        # Chi-square test
        contingency_table = pd.crosstab(data[feature_col], data[group_col])
        if contingency_table.min().min() >= 5:  # Expected frequency ≥5
            chi2, p_value, _, _ = chi2_contingency(contingency_table)
            test_method = 'Chi-square test'
        else:
            # Use Fisher's exact test
            if contingency_table.shape == (2, 2):
                _, p_value = fisher_exact(contingency_table)
                test_method = 'Fisher exact test'
            else:
                chi2, p_value, _, _ = chi2_contingency(contingency_table)
                test_method = 'Chi-square test'
    else:
        # Continuous variables: first test for normality
        _, p_norm_0 = stats.shapiro(group_0.sample(min(5000, len(group_0))))
        _, p_norm_1 = stats.shapiro(group_1.sample(min(5000, len(group_1))))
        
        if p_norm_0 > 0.05 and p_norm_1 > 0.05:
            # Normal distribution, use t-test
            _, p_value = ttest_ind(group_0, group_1)
            test_method = 'Independent t-test'
        else:
            # Non-normal distribution, use Mann-Whitney U test
            _, p_value = mannwhitneyu(group_0, group_1, alternative='two-sided')
            test_method = 'Mann-Whitney U test'
    
    return test_method, p_value

def calculate_statistics(data, group_col, feature_col, var_type):
    """Calculate descriptive statistics"""
    overall_data = data[feature_col].dropna()
    group_0_data = data[data[group_col] == 0][feature_col].dropna()
    group_1_data = data[data[group_col] == 1][feature_col].dropna()
    
    if var_type == 'Continuous':
        # Continuous variables: mean ± standard deviation
        overall_stats = f"{overall_data.mean():.2f}±{overall_data.std():.2f}"
        group_0_stats = f"{group_0_data.mean():.2f}±{group_0_data.std():.2f}"
        group_1_stats = f"{group_1_data.mean():.2f}±{group_1_data.std():.2f}"
    else:
        # Categorical variables: mode and proportion
        overall_mode = overall_data.mode()[0] if len(overall_data.mode()) > 0 else 'N/A'
        group_0_mode = group_0_data.mode()[0] if len(group_0_data.mode()) > 0 else 'N/A'
        group_1_mode = group_1_data.mode()[0] if len(group_1_data.mode()) > 0 else 'N/A'
        
        overall_prop = (overall_data == overall_mode).mean() if overall_mode != 'N/A' else 0
        group_0_prop = (group_0_data == group_0_mode).mean() if group_0_mode != 'N/A' else 0
        group_1_prop = (group_1_data == group_1_mode).mean() if group_1_mode != 'N/A' else 0
        
        overall_stats = f"{overall_mode} ({overall_prop:.2f})"
        group_0_stats = f"{group_0_mode} ({group_0_prop:.2f})"
        group_1_stats = f"{group_1_mode} ({group_1_prop:.2f})"
    
    return overall_stats, group_0_stats, group_1_stats

def get_range_string(data, feature_col, var_type):
    """Get variable range string"""
    feature_data = data[feature_col].dropna()
    
    if var_type == 'Continuous':
        min_val = feature_data.min()
        max_val = feature_data.max()
        return f"[{min_val:.2f}, {max_val:.2f}]"
    else:
        unique_vals = sorted(feature_data.unique())
        return "{" + ", ".join(map(str, unique_vals)) + "}"

def main():
    # Feature name mapping
    feature_name_mapping = {
        'age_at_visit': 'Age',
        'SEX': 'Sex',
        'EDUCYRS': 'Education Years',
        'duration': 'Disease Duration',
        'NHY': 'H&Y Stage',
        'updrs1_score': 'UPDRS-I',
        'updrs2_score': 'UPDRS-II',
        'updrs3_score': 'UPDRS-III',
        'updrs4_score': 'UPDRS-IV',
        'ess': 'ESS',
        'rem': 'RBDSQ',
        'gds': 'GDS'
    }
    
    # Read data
    file_path = 'PPMI_5_label.csv'
    data = pd.read_csv(file_path)
    
    if 'MCI' not in data.columns:
        print("Error: MCI column not found, please check the data file")
        return
    
    # Define feature columns to analyze (excluding target variable)
    feature_columns = [col for col in data.columns if col != 'MCI']
    
    print(f"Total data count: {len(data)}")
    print(f"PD-NC group (MCI=0): {len(data[data['MCI'] == 0])}")
    print(f"PD-CI group (MCI=1): {len(data[data['MCI'] == 1])}")
    print(f"Number of features to analyze: {len(feature_columns)}")
    print(f"Feature list: {feature_columns}")
    
    # Store results
    results = []
    p_values = []
    
    # Add sample size row
    results.append({
        'Variable': 'Sample Size',
        'Range': '',
        'Overall': len(data),
        'PD-NC Group': len(data[data['MCI'] == 0]),
        'PD-MCI group': len(data[data['MCI'] == 1]),
        'Variable Type': '',
        'Test Method': '',
        'P Value': '',
        'Adjusted P Value': ''
    })
    
    # Analyze each feature
    for feature in feature_columns:
        # Get display name
        display_name = feature_name_mapping.get(feature, feature)
        print(f"\nAnalyzing: {feature} ({display_name})")
        
        # Skip features with too many missing values
        missing_rate = data[feature].isnull().sum() / len(data)
        if missing_rate > 0.5:
            print(f"  Skipping {feature}: too many missing values ({missing_rate:.2%})")
            continue
        
        # Determine variable type
        var_type = determine_variable_type(data[feature])
        print(f"  Variable type: {var_type}")
        
        # Get range
        range_str = get_range_string(data, feature, var_type)
        
        # Calculate descriptive statistics
        overall_stats, group_0_stats, group_1_stats = calculate_statistics(
            data, 'MCI', feature, var_type
        )
        
        # Perform statistical test
        test_method, p_value = perform_statistical_test(
            data, 'MCI', feature, var_type
        )
        
        print(f"  Statistical test: {test_method}")
        print(f"  p-value: {p_value:.6f}")
        
        results.append({
            'Variable': display_name,
            'Range': range_str,
            'Overall': overall_stats,
            'PD-NC Group': group_0_stats,
            'PD-MCI group': group_1_stats,
            'Variable Type': var_type,
            'Test Method': test_method,
            'P Value': p_value,
            'Adjusted P Value': ''  # Will be filled after FDR correction
        })
        
        p_values.append(p_value)
    
    # FDR correction
    if p_values:
        print(f"\nPerforming FDR multiple comparison correction...")
        rejected, p_adjusted = fdrcorrection(p_values, alpha=0.05)
        
        # Update adjusted p-values
        for i, result in enumerate(results[1:]):  # Skip sample size row
            if p_adjusted[i] < 0.001:
                result['Adjusted P Value'] = '<0.001'
            else:
                result['Adjusted P Value'] = f'{p_adjusted[i]:.3f}'
            
            # Format original p-values
            if result['P Value'] < 0.001:
                result['P Value'] = '<0.001'
            else:
                result['P Value'] = f'{result["P Value"]:.3f}'
    
    # Save results
    results_df = pd.DataFrame(results)
    
    # Exclude columns 6, 7, 8 when saving (Variable Type, Test Method, P Value)
    columns_to_save = ['Variable', 'Range', 'Overall', 'PD-NC Group', 'PD-MCI group', 'Adjusted P Value']
    results_df_filtered = results_df[columns_to_save]
    
    output_file = '2_statistical_analysis.csv'
    results_df_filtered.to_csv(output_file, index=False, encoding='utf-8-sig')
    
    print(f"\nAnalysis completed! Results saved to: {output_file}")
    print(f"\nResults preview:")
    print(results_df_filtered.to_string(index=False))
    
    # Output summary of significant results
    print(f"\nSignificant results summary (FDR-corrected p < 0.05):")
    significant_results = results_df[1:][results_df[1:]['Adjusted P Value'] != ''].copy()
    significant_results = significant_results[significant_results['Adjusted P Value'] != 'N/A']
    
    for _, row in significant_results.iterrows():
        try:
            p_adj = float(row['Adjusted P Value']) if row['Adjusted P Value'] != '<0.001' else 0.0001
            if p_adj < 0.05:
                print(f"  {row['Variable']}: p_adj = {row['Adjusted P Value']} ({row['Test Method']})")
        except:
            continue
    
    return results_df

if __name__ == "__main__":
    results = main()
