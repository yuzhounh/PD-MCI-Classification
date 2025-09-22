import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import f_oneway, chi2_contingency
from sklearn.feature_selection import mutual_info_classif

# Set font to Arial for plots
plt.rcParams['font.sans-serif'] = ['Arial']

def load_and_prepare_data():
    """Load and prepare data"""
    # Read training data
    train_data = pd.read_csv('PPMI_6_train.csv')
    
    # Read feature mapping table
    try:
        mapping_data = pd.read_csv('PPMI_feature_mapping.csv')
        feature_mapping = dict(zip(mapping_data['Feature Name'], mapping_data['Abbreviation']))
    except:
        # If mapping file doesn't exist, use original column names
        feature_mapping = {}
        print("Warning: PPMI_feature_mapping.csv not found. Using original feature names.")
    
    # Remove subject ID column (first column) and extract target variable (last column)
    X = train_data.iloc[:, 1:-1]  # Feature variables
    y = train_data.iloc[:, -1]    # Target variable
    
    print(f"Data shape: {train_data.shape}")
    print(f"Feature variables shape: {X.shape}")
    print(f"Target variable distribution:\n{y.value_counts()}")
    
    return X, y, feature_mapping

def calculate_anova_f_score(X, y):
    """Calculate ANOVA F-test scores"""
    f_scores = []
    p_values = []
    
    for column in X.columns:
        # Group data by target variable
        group_0 = X[y == 0][column].dropna()
        group_1 = X[y == 1][column].dropna()
        
        if len(group_0) > 0 and len(group_1) > 0:
            f_stat, p_val = f_oneway(group_0, group_1)
            f_scores.append(f_stat)
            p_values.append(p_val)
        else:
            f_scores.append(0)
            p_values.append(1)
    
    return np.array(f_scores), np.array(p_values)

def calculate_chi2_score(X, y):
    """Calculate Chi-square test scores"""
    chi2_scores = []
    p_values = []
    
    for column in X.columns:
        # Bin continuous variables
        feature_data = X[column].dropna()
        if len(feature_data.unique()) > 10:  # If continuous variable
            # Use quartile binning
            feature_binned = pd.qcut(feature_data, q=4, duplicates='drop', labels=False)
        else:
            feature_binned = feature_data
        
        # Create contingency table
        try:
            contingency_table = pd.crosstab(feature_binned, y[feature_data.index])
            chi2_stat, p_val, _, _ = chi2_contingency(contingency_table)
            chi2_scores.append(chi2_stat)
            p_values.append(p_val)
        except:
            chi2_scores.append(0)
            p_values.append(1)
    
    return np.array(chi2_scores), np.array(p_values)

def calculate_pearson_correlation(X, y):
    """Calculate Pearson correlation coefficients"""
    correlations = []
    p_values = []
    
    for column in X.columns:
        feature_data = X[column].dropna()
        target_data = y[feature_data.index]
        
        if len(feature_data) > 1:
            corr, p_val = stats.pearsonr(feature_data, target_data)
            correlations.append(abs(corr))  # Use absolute value
            p_values.append(p_val)
        else:
            correlations.append(0)
            p_values.append(1)
    
    return np.array(correlations), np.array(p_values)

def calculate_mutual_information(X, y):
    """Calculate mutual information"""
    # Handle missing values
    X_filled = X.fillna(X.median())
    
    # Calculate mutual information
    mi_scores = mutual_info_classif(X_filled, y, random_state=42)
    
    return mi_scores

def create_feature_scores_table(X, y, feature_mapping):
    """Create feature scoring table"""
    # Calculate various scores
    anova_scores, anova_pvals = calculate_anova_f_score(X, y)
    chi2_scores, chi2_pvals = calculate_chi2_score(X, y)
    pearson_scores, pearson_pvals = calculate_pearson_correlation(X, y)
    mi_scores = calculate_mutual_information(X, y)
    
    # Create results DataFrame
    results = pd.DataFrame({
        'Feature_Name': X.columns,
        'ANOVA_F_Score': anova_scores,
        'ANOVA_P_Value': anova_pvals,
        'Chi2_Score': chi2_scores,
        'Chi2_P_Value': chi2_pvals,
        'Pearson_Correlation': pearson_scores,
        'Pearson_P_Value': pearson_pvals,
        'Mutual_Information': mi_scores
    })
    
    # Add mapped feature names
    results['Abbreviation'] = results['Feature_Name'].map(feature_mapping).fillna(results['Feature_Name'])
    
    # Reorder columns
    results = results[['Feature_Name', 'Abbreviation', 'ANOVA_F_Score', 'ANOVA_P_Value', 
                      'Chi2_Score', 'Chi2_P_Value', 'Pearson_Correlation', 'Pearson_P_Value', 
                      'Mutual_Information']]
    
    return results

def normalize_and_average_scores(results):
    """Normalize scores using max normalization and calculate average"""
    # Extract scores from four scoring methods
    score_columns = ['ANOVA_F_Score', 'Chi2_Score', 'Pearson_Correlation', 'Mutual_Information']
    
    # Create normalized DataFrame
    normalized_scores = pd.DataFrame()
    normalized_scores['Feature_Name'] = results['Feature_Name']
    normalized_scores['Abbreviation'] = results['Abbreviation']
    
    # Normalize each method using max normalization
    for col in score_columns:
        max_val = results[col].max()
        if max_val > 0:
            normalized_scores[f'{col}_normalized'] = results[col] / max_val
        else:
            normalized_scores[f'{col}_normalized'] = 0
    
    # Calculate average importance across four methods
    normalized_columns = [f'{col}_normalized' for col in score_columns]
    normalized_scores['Average_Importance'] = normalized_scores[normalized_columns].mean(axis=1)
    
    return normalized_scores

def calculate_rank_based_average(results):
    """Calculate average ranking based on ranks"""
    # Extract scores from four scoring methods
    score_columns = ['ANOVA_F_Score', 'Chi2_Score', 'Pearson_Correlation', 'Mutual_Information']
    
    # Create ranking DataFrame
    rank_scores = pd.DataFrame()
    rank_scores['Feature_Name'] = results['Feature_Name']
    rank_scores['Abbreviation'] = results['Abbreviation']
    
    # Calculate ranks for each method (higher scores get better ranks, i.e., smaller rank numbers)
    for col in score_columns:
        # Use rank method, ascending=False means higher scores get smaller rank numbers (better ranks)
        rank_scores[f'{col}_rank'] = results[col].rank(ascending=False, method='min')
    
    # Calculate average rank across four methods
    rank_columns = [f'{col}_rank' for col in score_columns]
    rank_scores['Average_Rank'] = rank_scores[rank_columns].mean(axis=1)
    
    # Calculate importance score based on average rank (smaller rank = higher importance)
    # Use (total_features + 1 - average_rank) / total_features to convert to 0-1 importance score
    n_features = len(results)
    rank_scores['Rank_Based_Importance'] = (n_features + 1 - rank_scores['Average_Rank']) / n_features
    
    return rank_scores

def plot_feature_scores(results):
    """Plot feature scoring results in 2x2 grid"""
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    
    # Get feature abbreviations for display
    feature_names = results['Abbreviation'].tolist()
    
    # 1. ANOVA F-Score
    ax1 = axes[0, 0]
    scores = results['ANOVA_F_Score'].values
    sorted_idx = np.argsort(scores)
    sorted_features = [feature_names[i] for i in sorted_idx]
    sorted_scores = scores[sorted_idx]
    
    bars1 = ax1.barh(range(len(sorted_scores)), sorted_scores, color='#008BFB')
    ax1.set_yticks(range(len(sorted_scores)))
    ax1.set_yticklabels(sorted_features, fontsize=10)
    ax1.set_title('ANOVA F-Score', fontsize=12, fontweight='bold')
    ax1.set_xlabel('F-Score', fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # 2. Chi-Square Score
    ax2 = axes[0, 1]
    scores = results['Chi2_Score'].values
    sorted_idx = np.argsort(scores)
    sorted_features = [feature_names[i] for i in sorted_idx]
    sorted_scores = scores[sorted_idx]
    
    bars2 = ax2.barh(range(len(sorted_scores)), sorted_scores, color='#FF6B6B')
    ax2.set_yticks(range(len(sorted_scores)))
    ax2.set_yticklabels(sorted_features, fontsize=10)
    ax2.set_title('Chi-Square Score', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Chi2 Score', fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    # 3. Pearson Correlation
    ax3 = axes[1, 0]
    scores = results['Pearson_Correlation'].values
    sorted_idx = np.argsort(scores)
    sorted_features = [feature_names[i] for i in sorted_idx]
    sorted_scores = scores[sorted_idx]
    
    bars3 = ax3.barh(range(len(sorted_scores)), sorted_scores, color='#4ECDC4')
    ax3.set_yticks(range(len(sorted_scores)))
    ax3.set_yticklabels(sorted_features, fontsize=10)
    ax3.set_title('Pearson Correlation (Absolute)', fontsize=12, fontweight='bold')
    ax3.set_xlabel('|Correlation|', fontsize=11)
    ax3.grid(True, alpha=0.3)
    
    # 4. Mutual Information
    ax4 = axes[1, 1]
    scores = results['Mutual_Information'].values
    sorted_idx = np.argsort(scores)
    sorted_features = [feature_names[i] for i in sorted_idx]
    sorted_scores = scores[sorted_idx]
    
    bars4 = ax4.barh(range(len(sorted_scores)), sorted_scores, color='#FFD93D')
    ax4.set_yticks(range(len(sorted_scores)))
    ax4.set_yticklabels(sorted_features, fontsize=10)
    ax4.set_title('Mutual Information', fontsize=12, fontweight='bold')
    ax4.set_xlabel('MI Score', fontsize=11)
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure
    plt.savefig('5_filters_comparison.pdf', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Feature scoring results chart saved as:")
    print("- 5_filters_comparison.pdf")

def plot_average_importance(rank_scores):
    """Plot importance bar chart based on average ranking"""
    # Sort by average rank (smaller rank is more important, so descending order)
    sorted_data = rank_scores.sort_values('Average_Rank', ascending=False)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(5, 4))
    
    # Plot horizontal bar chart using rank-based importance scores
    bars = ax.barh(range(len(sorted_data)), 
                   sorted_data['Rank_Based_Importance'], 
                   color='#008BFB',
                   edgecolor='white',
                   linewidth=0.5,
                   alpha=0.8)
    
    # Set y-axis labels
    ax.set_yticks(range(len(sorted_data)))
    ax.set_yticklabels(sorted_data['Abbreviation'], fontsize=10)
    
    # Set title and labels
    ax.set_xlabel('Average Rank', fontsize=10)
    
    # Add grid
    ax.grid(True, alpha=0.3, axis='x')
    
    # Set xlim
    ax.set_xlim(0, 1.1)
    
    # Add average rank labels on bars
    for i, (bar, avg_rank, importance) in enumerate(zip(bars, sorted_data['Average_Rank'], sorted_data['Rank_Based_Importance'])):
        ax.text(importance + 0.02, i, f'{avg_rank:.1f}', 
               va='center', ha='left', fontsize=9)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure
    plt.savefig('5_filters_average_importance.pdf', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Average importance chart saved as:")
    print("- 5_filters_average_importance.pdf")
    
    return sorted_data

def create_detailed_table(rank_scores):
    """Create detailed ranking analysis table"""
    # Reorder columns
    columns_order = ['Feature_Name', 'Abbreviation', 
                    'ANOVA_F_Score_rank', 'Chi2_Score_rank', 
                    'Pearson_Correlation_rank', 'Mutual_Information_rank',
                    'Average_Rank', 'Rank_Based_Importance']
    
    detailed_table = rank_scores[columns_order].copy()
    
    # Rename columns for better display
    detailed_table.columns = ['Feature_Name', 'Abbreviation', 
                             'ANOVA_Rank', 'Chi2_Rank', 
                             'Pearson_Rank', 'MI_Rank',
                             'Average_Rank', 'Rank_Based_Importance']
    
    # Sort by average rank in ascending order (smaller rank is more important)
    detailed_table = detailed_table.sort_values('Average_Rank', ascending=True)
    
    return detailed_table

def main():
    """Main function"""
    print("Starting feature selection analysis...")
    
    # Load data
    X, y, feature_mapping = load_and_prepare_data()
    
    # Calculate feature scores
    print("\nCalculating feature scores...")
    results = create_feature_scores_table(X, y, feature_mapping)
    
    # Save results table
    results.to_csv('5_filters_comparison.csv', index=False)
    print(f"\nFeature scoring results saved as: 5_filters_comparison.csv")
    
    # Display results table
    print("\nFeature scoring results:")
    print("="*100)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    print(results.round(4))
    
    # Plot method-specific charts
    print("\nGenerating method-specific visualization charts...")
    plot_feature_scores(results)
    
    # Normalize and average
    print("\nPerforming max normalization and average calculation...")
    normalized_scores = normalize_and_average_scores(results)
    
    # Calculate rank-based average importance
    print("\nCalculating rank-based average importance...")
    rank_scores = calculate_rank_based_average(results)
    
    # Create detailed table
    print("Creating detailed analysis table...")
    detailed_table = create_detailed_table(rank_scores)
    
    # Save detailed table
    detailed_table.to_csv('5_filters_average_importance_detailed.csv', index=False)
    print("Detailed analysis results saved as: 5_filters_average_importance_detailed.csv")
    
    # Display detailed table
    print("\nDetailed feature average ranking analysis:")
    print("="*120)
    pd.set_option('display.precision', 4)
    print(detailed_table.round(4))
    
    # Plot rank-based average importance bar chart
    print("\nGenerating rank-based average importance bar chart...")
    sorted_data = plot_average_importance(rank_scores)
    
    # Display top 5 features for each scoring method
    print("\nTop 5 features for each scoring method:")
    print("-"*50)
    
    top5_anova = results.nlargest(5, 'ANOVA_F_Score')[['Abbreviation', 'ANOVA_F_Score']]
    print("ANOVA F-Score Top 5:")
    print(top5_anova.to_string(index=False))
    
    print("\nChi-Square Score Top 5:")
    top5_chi2 = results.nlargest(5, 'Chi2_Score')[['Abbreviation', 'Chi2_Score']]
    print(top5_chi2.to_string(index=False))
    
    print("\nPearson Correlation Top 5:")
    top5_pearson = results.nlargest(5, 'Pearson_Correlation')[['Abbreviation', 'Pearson_Correlation']]
    print(top5_pearson.to_string(index=False))
    
    print("\nMutual Information Top 5:")
    top5_mi = results.nlargest(5, 'Mutual_Information')[['Abbreviation', 'Mutual_Information']]
    print(top5_mi.to_string(index=False))
    
    # Display top 10 features based on average ranking
    print("\nTop 10 features based on average ranking:")
    print("-"*60)
    top_10 = detailed_table.head(10)[['Abbreviation', 'Average_Rank', 'Rank_Based_Importance']]
    for i, (_, row) in enumerate(top_10.iterrows(), 1):
        print(f"{i:2d}. {row['Abbreviation']:12s} Average Rank: {row['Average_Rank']:5.1f} Importance: {row['Rank_Based_Importance']:.4f}")
    
    print(f"\nAll charts saved:")
    print("- 5_filters_comparison.pdf (method-specific charts)")
    print("- 5_filters_average_importance.pdf (rank-based average importance chart)")
    print("- 5_filters_average_importance_detailed.csv (detailed analysis table)")
    
    print("\nAnalysis completed!")
    
    # Display method description
    print("\nMethod description:")
    print("-"*60)
    print("1. Calculate feature rankings for four filter methods separately")
    print("2. ANOVA F-Score: Analysis of variance F-test")
    print("3. Chi2 Score: Chi-square test")
    print("4. Pearson Correlation: Pearson correlation coefficient (absolute value)")
    print("5. Mutual Information: Mutual information")
    print("6. Calculate arithmetic mean of rankings from four methods")
    print("7. Calculate importance score based on average ranking: (total_features + 1 - average_rank) / total_features")

if __name__ == "__main__":
    main()