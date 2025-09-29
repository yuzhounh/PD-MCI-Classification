import pandas as pd

def analyze_patno_site_consistency(csv_file):
    """
    Analyze whether the same subject (PATNO) is always at the same site (SITE) in the CSV file
    
    Parameters:
    csv_file: CSV file path
    
    Returns:
    Analysis results dictionary
    """
    
    # Read CSV file
    try:
        df = pd.read_csv(csv_file)
        print(f"Successfully read file: {csv_file}")
        print(f"Data dimensions: {df.shape}")
    except Exception as e:
        print(f"Failed to read file: {e}")
        return None
    
    # Check if required columns exist
    required_columns = ['SITE', 'PATNO']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        print(f"Missing required columns: {missing_columns}")
        return None
    
    print(f"Columns in file: {list(df.columns)}")
    print("\n" + "="*50)
    
    # Group by PATNO and check if SITE is unique for each subject
    patno_site_groups = df.groupby('PATNO')['SITE'].agg(['nunique', 'unique', 'count']).reset_index()
    patno_site_groups.columns = ['PATNO', 'site_count', 'sites', 'record_count']
    
    # Find subjects with inconsistent sites
    inconsistent_patno = patno_site_groups[patno_site_groups['site_count'] > 1]
    
    # Statistical results
    total_patients = len(patno_site_groups)
    consistent_patients = len(patno_site_groups[patno_site_groups['site_count'] == 1])
    inconsistent_patients = len(inconsistent_patno)
    
    print("Data Overview:")
    print(f"Total number of subjects: {total_patients}")
    print(f"Total number of records: {len(df)}")
    print(f"Subjects with consistent sites: {consistent_patients}")
    print(f"Subjects with inconsistent sites: {inconsistent_patients}")
    print(f"Consistency rate: {consistent_patients/total_patients*100:.2f}%")
    
    # Show site distribution
    print(f"\nSite Distribution:")
    site_counts = df['SITE'].value_counts().sort_index()
    for site, count in site_counts.items():
        print(f"Site {site}: {count} records")
    
    # If there are inconsistent subjects, show detailed information
    if inconsistent_patients > 0:
        print(f"\nDetails of subjects with inconsistent sites:")
        for _, row in inconsistent_patno.iterrows():
            patno = row['PATNO']
            sites = row['sites']
            record_count = row['record_count']
            print(f"Subject {patno}: appears at sites {sites}, total {record_count} records")
        
        # Show detailed records of these inconsistent subjects
        print(f"\nSpecific records of inconsistent subjects:")
        for patno in inconsistent_patno['PATNO']:
            print(f"\nAll records for subject {patno}:")
            patient_records = df[df['PATNO'] == patno][['SITE', 'PATNO', 'age_at_visit']].sort_values('age_at_visit')
            print(patient_records.to_string(index=False))
    else:
        print(f"\n✅ All subjects have data collected at the same site, good site consistency!")
    
    # Return analysis results
    results = {
        'total_patients': total_patients,
        'consistent_patients': consistent_patients,
        'inconsistent_patients': inconsistent_patients,
        'consistency_rate': consistent_patients/total_patients,
        'inconsistent_patno_list': inconsistent_patno['PATNO'].tolist() if inconsistent_patients > 0 else [],
        'site_distribution': site_counts.to_dict()
    }
    
    return results

def detailed_inconsistency_analysis(csv_file):
    """
    Perform more detailed analysis of site inconsistencies
    """
    df = pd.read_csv(csv_file)
    
    # Find all inconsistent PATNOs
    patno_site_groups = df.groupby('PATNO')['SITE'].nunique().reset_index()
    patno_site_groups.columns = ['PATNO', 'site_count']
    inconsistent_patnos = patno_site_groups[patno_site_groups['site_count'] > 1]['PATNO'].tolist()
    
    if not inconsistent_patnos:
        print("No subjects with site inconsistencies found")
        return
    
    print("Detailed Inconsistency Analysis:")
    print("="*60)
    
    for patno in inconsistent_patnos:
        patient_data = df[df['PATNO'] == patno].sort_values('age_at_visit')
        sites = patient_data['SITE'].unique()
        
        print(f"\nSubject {patno}:")
        print(f"Sites involved: {sites}")
        print(f"Number of visits: {len(patient_data)}")
        print(f"Age range: {patient_data['age_at_visit'].min():.2f} - {patient_data['age_at_visit'].max():.2f}")
        
        # Show grouped by site
        for site in sites:
            site_data = patient_data[patient_data['SITE'] == site]
            print(f"  Site {site}: {len(site_data)} visits")
            if 'age_at_visit' in site_data.columns:
                ages = site_data['age_at_visit'].values
                print(f"    Ages: {', '.join([f'{age:.2f}' for age in ages])}")

# Main program execution
if __name__ == "__main__":
    csv_file = "PPMI_5_label.csv"
    
    print("Starting subject site consistency analysis...")
    print("="*50)
    
    # Execute basic analysis
    results = analyze_patno_site_consistency(csv_file)
    
    if results and results['inconsistent_patients'] > 0:
        print("\n" + "="*50)
        # If inconsistencies found, execute detailed analysis
        detailed_inconsistency_analysis(csv_file)
    
    print(f"\nAnalysis completed!")
