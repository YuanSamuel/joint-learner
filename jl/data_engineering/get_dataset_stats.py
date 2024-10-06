import os
import pandas as pd
import sys

# Function to compute statistics and append to a summary CSV if dataset_name doesn't exist
def process_csv_and_append_statistics(csv_file, output_csv='data/statistics_summary.csv'):
    # Extract dataset_name from the file path
    dataset_name = os.path.basename(csv_file).replace('.csv', '')
    
    # Read the CSV file into a DataFrame
    df = pd.read_csv(csv_file)

    # Compute statistics
    not_cached_count = df['decision'].value_counts().get('Not Cached', 0)
    total_requests = len(df)
    not_cached_percentage = (not_cached_count / total_requests) * 100 if total_requests > 0 else 0

    # Count unique modified full_addr values
    df['modified_full_addr'] = (df['full_addr'] // 64) * 64
    unique_full_addrs_count = df['modified_full_addr'].nunique()

    # Count unique IPs (PCs)
    unique_ip_count = df['ip'].nunique()

    # Prepare statistics as a dictionary
    stats = {
        'dataset_name': dataset_name,
        'not_cached_count': not_cached_count,
        'total_requests': total_requests,
        'not_cached_percentage': not_cached_percentage,
        'unique_full_addrs_count': unique_full_addrs_count,
        'unique_ip_count': unique_ip_count
    }

    # Convert dictionary to DataFrame
    stats_df = pd.DataFrame([stats])

    # Check if output CSV exists, if not create it
    if not os.path.exists(output_csv):
        stats_df.to_csv(output_csv, index=False)
        print(f"Created new summary file: {output_csv}")
    else:
        # Load existing summary CSV
        summary_df = pd.read_csv(output_csv)

        # Check if dataset_name already exists
        if dataset_name not in summary_df['dataset_name'].values:
            # Append new statistics and save to CSV
            updated_df = pd.concat([summary_df, stats_df], ignore_index=True)
            updated_df.to_csv(output_csv, index=False)
            print(f"Appended statistics for {dataset_name}")
        else:
            print(f"Dataset {dataset_name} already exists in the summary CSV. Skipping.")

# Main method to take CSV file path as input
if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python script.py <csv_file_path>")
    else:
        csv_file_path = sys.argv[1]
        process_csv_and_append_statistics(csv_file_path)
