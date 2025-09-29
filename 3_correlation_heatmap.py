import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages

# Set font to Arial for plots
plt.rcParams['font.sans-serif'] = ['Arial']

# Read data file
data = pd.read_csv('PPMI_5_label.csv')
feature_mapping = pd.read_csv('PPMI_feature_mapping.csv')

print(f"Data shape: {data.shape}")
print(f"Feature mapping table shape: {feature_mapping.shape}")

# Create feature name mapping dictionary
name_mapping = dict(zip(feature_mapping['Feature Name'], feature_mapping['Abbreviation']))
print(f"Feature mapping dictionary: {name_mapping}")

# Get feature columns to analyze (excluding first and last columns)
feature_columns = data.columns[1:-1]
print(f"Feature columns to analyze: {list(feature_columns)}")

# Extract feature data
feature_data = data[feature_columns]

# Apply feature name mapping
mapped_columns = []
for col in feature_columns:
    if col in name_mapping:
        mapped_columns.append(name_mapping[col])
    else:
        mapped_columns.append(col)  # Keep original name if no mapping exists

# Rename columns
feature_data.columns = mapped_columns
print(f"Mapped feature names: {list(feature_data.columns)}")

# Calculate correlation matrix
correlation_matrix = feature_data.corr()
print("Correlation matrix:")
print(correlation_matrix)

# Create mask: hide upper triangle and diagonal
mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))

# # Create mask: hide only diagonal
# mask = np.eye(correlation_matrix.shape[0], dtype=bool)

# Create heatmap
plt.figure(figsize=(7, 7))

# Use seaborn to create heatmap
sns.heatmap(correlation_matrix, 
            mask=mask,            # Apply mask
            annot=True,           # Show correlation coefficient values
            fmt='.2f',            # Keep 2 decimal places
            cmap='coolwarm',      # Color mapping
            center=0,             # Center at 0
            square=True,          # Square cells
            linewidths=0.5,       # Grid line width
            cbar_kws={'shrink': 0.6, 'aspect': 20},  # Shrink color bar
            # annot_kws={'size': 8}, # Set font size for correlation values
            )     

# Set font size for x-axis and y-axis labels
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)

# Adjust layout
plt.tight_layout()

# Save as PDF file
with PdfPages('3_correlation_heatmap.pdf') as pdf:
    pdf.savefig(plt.gcf(), bbox_inches='tight', dpi=300)
    
print("Correlation heatmap saved as '3_correlation_heatmap.pdf'")

# Don't display figure
plt.show()  # Comment out this line

# Output some statistics
print("\nCorrelation Statistics:")
print(f"Highest positive correlation: {correlation_matrix.values[np.triu_indices_from(correlation_matrix.values, k=1)].max():.3f}")
print(f"Lowest negative correlation: {correlation_matrix.values[np.triu_indices_from(correlation_matrix.values, k=1)].min():.3f}")

# Find highly correlated feature pairs
high_corr_threshold = 0.7
high_correlations = []
for i in range(len(correlation_matrix.columns)):
    for j in range(i+1, len(correlation_matrix.columns)):
        corr_value = correlation_matrix.iloc[i, j]
        if abs(corr_value) > high_corr_threshold:
            high_correlations.append((
                correlation_matrix.columns[i], 
                correlation_matrix.columns[j], 
                corr_value
            ))

if high_correlations:
    print(f"\nHigh correlation feature pairs (|r| > {high_corr_threshold}):")
    for feat1, feat2, corr in high_correlations:
        print(f"{feat1} - {feat2}: {corr:.3f}")
else:
    print(f"\nNo high correlation feature pairs found (|r| > {high_corr_threshold})")