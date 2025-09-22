import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Set font to Arial for plots
plt.rcParams['font.sans-serif'] = ['Arial']

# Data definition
features = ['Age', 'EDUCYRS', 'Duration', 'UPDRS-I', 'GDS', 'UPDRS-III', 
           'Sex', 'ESS', 'UPDRS-II', 'UPDRS-IV', 'RBDSQ', 'H&Y']

methods = ['ANOVA', 'Chi-Square', 'Pearson', 'MI', 'LR', 'SVM', 'RF', 'XGBoost',
          'LR', 'SVM', 'RF', 'XGBoost', 'LR', 'SVM', 'RF', 'XGBoost', 
          'LR', 'SVM', 'RF', 'XGBoost']

method_groups = ['Filters', 'Filters', 'Filters', 'Filters', 
                'Wrappers-RFE', 'Wrappers-RFE', 'Wrappers-RFE', 'Wrappers-RFE',
                'Wrappers-SFS', 'Wrappers-SFS', 'Wrappers-SFS', 'Wrappers-SFS',
                'Wrappers-SBE', 'Wrappers-SBE', 'Wrappers-SBE', 'Wrappers-SBE',
                'Embedded', 'Embedded', 'Embedded', 'Embedded']

data = np.array([
    [1, 1, 1, 1, 1, 2, 1, 2, 2, 4, 3, 2, 2, 3, 2, 2, 1, 2, 1, 2],
    [2, 2, 2, 2, 2, 1, 5, 1, 1, 3, 1, 1, 1, 1, 1, 1, 2, 1, 4, 1],
    [5, 7, 5, 3, 3, 6, 2, 3, 4, 6, 5, 6, 4, 11, 3, 5, 3, 3, 2, 6],
    [4, 4, 4, 6, 5, 4, 6, 7, 5, 9, 6, 9, 10, 2, 9, 9, 5, 4, 6, 8],
    [3, 3, 3, 9, 4, 4, 9, 5, 3, 1, 10, 11, 3, 10, 7, 6, 4, 7, 9, 3],
    [9, 5, 9, 4, 7, 6, 3, 9, 12, 7, 12, 12, 12, 7, 8, 10, 7, 8, 3, 11],
    [12, 12, 12, 9, 6, 3, 12, 4, 9, 8, 2, 5, 6, 12, 6, 3, 6, 5, 12, 4],
    [10, 11, 10, 8, 9, 8, 7, 8, 10, 5, 11, 10, 8, 5, 12, 12, 9, 6, 7, 9],
    [6, 9, 6, 5, 12, 10, 4, 12, 7, 12, 4, 8, 5, 6, 10, 4, 10, 9, 5, 12],
    [7, 10, 7, 9, 10, 9, 10, 6, 8, 2, 9, 7, 11, 8, 11, 8, 12, 12, 10, 7],
    [8, 8, 8, 9, 10, 11, 8, 11, 6, 11, 7, 4, 7, 9, 5, 11, 11, 11, 8, 10],
    [11, 6, 11, 7, 8, 11, 11, 10, 11, 10, 8, 3, 9, 4, 4, 7, 8, 10, 11, 5]
])

def plot_original_order():
    """
    Plot original order heatmap
    """
    fig, ax = plt.subplots(figsize=(10, 7))
    
    df = pd.DataFrame(data, index=features, columns=methods)
    
    im = ax.imshow(df.values, cmap='GnBu_r', aspect='auto', vmin=1, vmax=12)
    # ax.set_title('Original Order', fontsize=16, fontweight='bold')
    ax.set_xticks(np.arange(len(methods)))
    ax.set_yticks(np.arange(len(features)))
    ax.set_xticklabels(methods, rotation=45, ha='right', fontsize=10)
    ax.set_yticklabels(features, fontsize=10)
    
    # Remove image borders
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    
    # Add white grid lines
    ax.set_xticks(np.arange(len(methods)+1) - 0.5, minor=True)
    ax.set_yticks(np.arange(len(features)+1) - 0.5, minor=True)
    ax.grid(which='minor', color='white', linestyle='-', linewidth=1)
    
    # Add values
    for i in range(df.shape[0]):
        for j in range(df.shape[1]):
            value = df.iloc[i, j]
            color = 'white' if value <= 6 else 'black'
            ax.text(j, i, str(int(value)), ha='center', va='center', 
                   color=color, fontsize=9)
    
    # Add method group annotations
    unique_groups = []
    group_positions = []
    current_group = None
    start_idx = 0
    
    for i, group in enumerate(method_groups):
        if group != current_group:
            if current_group is not None:
                # Calculate center position of previous group
                center_pos = (start_idx + i - 1) / 2
                unique_groups.append(current_group)
                group_positions.append(center_pos)
            current_group = group
            start_idx = i
    
    # Handle the last group
    if current_group is not None:
        center_pos = (start_idx + len(method_groups) - 1) / 2
        unique_groups.append(current_group)
        group_positions.append(center_pos)
    
    # Add group labels below x-axis, with appropriate offset
    for group, pos in zip(unique_groups, group_positions):
        ax.text(pos, len(features) + 0.9, group, ha='center', va='top', 
               fontsize=12)
    
    # # Add colorbar
    # cbar = plt.colorbar(im, ax=ax)
    # cbar.set_label('Feature Ranking (1=Most Important)', rotation=270, labelpad=20, fontsize=12)
    # cbar.ax.invert_yaxis()
    
    plt.tight_layout()
    return fig, ax

# Main usage example
if __name__ == "__main__":
    # Create original order plot
    print("Creating original order plot...")
    fig1, ax1 = plot_original_order()
    plt.savefig('10_overall_ranking.pdf', dpi=300, bbox_inches='tight')
    # plt.savefig('10_overall_ranking.png', dpi=300, bbox_inches='tight')
    plt.show()