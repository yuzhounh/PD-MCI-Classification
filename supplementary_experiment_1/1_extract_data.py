import pandas as pd
import numpy as np

# 0. Read the Excel file
print("Step 0: Reading the Excel file")
df = pd.read_excel("../PPMI_Curated_Data_Cut_Public_20250321.xlsx")
print(f"Original data shape: {df.shape}")

# 1. Select rows where COHORT = 1
print("\nStep 1: Selecting rows where COHORT = 1")
df_pd = df[df['COHORT'] == 1]
df_pd.to_csv("PPMI_1_PD.csv", index=False)
print(f"PPMI_1_PD.csv shape: {df_pd.shape}")

# 2. Extract specified features
print("\nStep 2: Extracting specified features")
features = [
    'SITE',
    'PATNO',
    'age_at_visit',
    'SEX',
    'EDUCYRS',
    'ageonset',
    'NHY',
    'updrs1_score',
    'updrs2_score',
    'updrs3_score',
    'updrs4_score',
    'ess',
    'moca',
    'rem',
    'gds'
]
df_features = df_pd[features]
df_features.to_csv("PPMI_2_features.csv", index=False)
print(f"PPMI_2_features.csv shape: {df_features.shape}")

# 3. Calculate duration and replace ageonset column
print("\nStep 3: Calculating duration")
df_duration = df_features.copy()
df_duration['duration'] = df_duration['age_at_visit'] - df_duration['ageonset']
df_duration = df_duration.drop(columns=['ageonset'])
# Move the 'duration' column to the fifth position
df_duration = df_duration[[col for col in df_duration.columns[:4]] + ['duration'] + [col for col in df_duration.columns[4:] if col != 'duration']]
df_duration.to_csv("PPMI_3_calculate_duration.csv", index=False)
print(f"PPMI_3_calculate_duration.csv shape: {df_duration.shape}")

# 4. Remove rows with missing values
print("\nStep 4: Removing rows with missing values")
df_clean = df_duration.dropna()
df_clean.to_csv("PPMI_4_remove_nan.csv", index=False)
print(f"PPMI_4_remove_nan.csv shape: {df_clean.shape}")

# 5. Calculate MCI label
print("\nStep 5: Calculating MCI label")
df_label = df_clean.copy()
# Create MCI column
df_label['MCI'] = np.nan
df_label.loc[df_label['moca'] > 25, 'MCI'] = 0
df_label.loc[(df_label['moca'] >= 21) & (df_label['moca'] <= 25), 'MCI'] = 1

# Remove samples with moca < 21
df_label = df_label[df_label['moca'] >= 21]

# Count samples in each class
mci_0_count = sum(df_label['MCI'] == 0)
mci_1_count = sum(df_label['MCI'] == 1)
print(f"Number of samples with MCI = 0: {mci_0_count}")
print(f"Number of samples with MCI = 1: {mci_1_count}")

# Remove moca column and move MCI to the rightmost position
df_label = df_label.drop(columns=['moca'])
mci_values = df_label.pop('MCI')
df_label['MCI'] = mci_values

df_label.to_csv("PPMI_5_label.csv", index=False)
print(f"PPMI_5_label.csv shape: {df_label.shape}")
