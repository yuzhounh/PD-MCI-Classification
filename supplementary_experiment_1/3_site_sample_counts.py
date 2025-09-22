import pandas as pd
import numpy as np

def analyze_ppmi_data(csv_file_path):
    """
    Read PPMI data and count the number of samples at each site
    
    Parameters:
    csv_file_path: Path to the CSV file
    """
    try:
        # Read CSV file
        df = pd.read_csv(csv_file_path)
        print(f"Successfully read data file: {csv_file_path}")
        print(f"Total number of rows: {len(df)}")
        print(f"Total number of columns: {len(df.columns)}")
        print()
        
        # Display basic data information
        print("First 5 rows of data:")
        print(df.head())
        print()
        
        # Count the number of samples at each site
        site_counts = df['SITE'].value_counts().sort_index()
        
        print("Sample count statistics by site:")
        print("=" * 30)
        print(f"{'Site ID':<8} {'Sample Count':<12}")
        print("-" * 22)
        
        total_samples = 0
        for site_id, count in site_counts.items():
            print(f"{site_id:<8} {count:<12}")
            total_samples += count
        
        print("-" * 22)
        print(f"{'Total':<8} {total_samples:<12}")
        print()
        
        # Display statistical summary
        print("Statistical Summary:")
        print(f"Total number of sites: {len(site_counts)}")
        print(f"Average samples per site: {total_samples / len(site_counts):.2f}")
        print(f"Site with most samples: {site_counts.idxmax()} (Sample count: {site_counts.max()})")
        print(f"Site with least samples: {site_counts.idxmin()} (Sample count: {site_counts.min()})")
        
        # Generate detailed report
        print("\nDetailed Statistical Report:")
        print(site_counts.describe())
        
        return site_counts
        
    except FileNotFoundError:
        print(f"Error: File '{csv_file_path}' not found")
        print("Please ensure the file path is correct and the file exists.")
        return None
    except Exception as e:
        print(f"Error occurred while reading file: {e}")
        return None

# Usage example
if __name__ == "__main__":
    # Specify CSV file path
    file_path = "PPMI_5_label.csv"
    
    # Execute analysis
    site_statistics = analyze_ppmi_data(file_path)
    
    # Save results to new file if needed
    if site_statistics is not None:
        # Save statistical results to CSV file
        site_statistics.to_csv("2_site_sample_counts.csv", header=['Sample Count'])
        print(f"\nStatistical results saved to: 2_site_sample_counts.csv")

    # Remove PATNO column and save results to new CSV file
    df = pd.read_csv(file_path)
    df.drop(columns=['PATNO'], inplace=True)
    df.to_csv('PPMI_5_remove_PATNO.csv', index=False)
    print(f"\nRemoved PATNO column from table and saved to: PPMI_5_remove_PATNO.csv")