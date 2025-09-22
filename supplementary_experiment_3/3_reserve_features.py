import pandas as pd
import os

def process_csv_files():
    """
    Read PPMI_6_train.csv and PPMI_6_test.csv files,
    keep only specified columns and save as new files
    """
    # Define columns to keep
    columns_to_keep = ['PATNO', 'age_at_visit', 'EDUCYRS', 'duration', 'updrs1_score', 'gds', 'MCI']
    
    # Input and output file configuration
    files_config = [
        {
            'input': 'PPMI_6_train.csv',
            'output': 'PPMI_7_train.csv'
        },
        {
            'input': 'PPMI_6_test.csv',
            'output': 'PPMI_7_test.csv'
        }
    ]
    
    # Process each file
    for config in files_config:
        input_file = config['input']
        output_file = config['output']
        
        try:
            # Check if input file exists
            if not os.path.exists(input_file):
                print(f"Error: File {input_file} does not exist")
                continue
            
            # Read CSV file
            print(f"Reading {input_file}...")
            df = pd.read_csv(input_file)
            
            # Check if all required columns exist
            missing_columns = [col for col in columns_to_keep if col not in df.columns]
            if missing_columns:
                print(f"Warning: File {input_file} is missing the following columns: {missing_columns}")
                # Keep only existing columns
                available_columns = [col for col in columns_to_keep if col in df.columns]
                df_filtered = df[available_columns]
            else:
                # Keep only specified columns
                df_filtered = df[columns_to_keep]
            
            # Save to new file
            df_filtered.to_csv(output_file, index=False)
            print(f"Successfully processed: {input_file} -> {output_file}")
            print(f"Original data shape: {df.shape}")
            print(f"Processed data shape: {df_filtered.shape}")
            print(f"Columns kept: {list(df_filtered.columns)}")
            print("-" * 50)
            
        except FileNotFoundError:
            print(f"Error: File {input_file} not found")
        except pd.errors.EmptyDataError:
            print(f"Error: File {input_file} is empty")
        except Exception as e:
            print(f"Error processing file {input_file}: {str(e)}")

def preview_results():
    """
    Preview the processed file contents
    """
    output_files = ['PPMI_7_train.csv', 'PPMI_7_test.csv']
    
    for file in output_files:
        if os.path.exists(file):
            print(f"\nPreviewing file: {file}")
            df = pd.read_csv(file)
            print(f"Data shape: {df.shape}")
            print(f"Column names: {list(df.columns)}")
            print("First 5 rows:")
            print(df.head())
            print("-" * 50)
        else:
            print(f"File {file} does not exist")

if __name__ == "__main__":
    # Execute file processing
    process_csv_files()
    
    # Preview results
    print("\n" + "="*50)
    print("Processing completed, previewing results:")
    preview_results()