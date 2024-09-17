import pandas as pd

# Replace 'your_file.csv' with the path to your CSV file
csv_file = 'data/mcf_large_labeled.csv'

# Read the CSV file into a DataFrame
df = pd.read_csv(csv_file)

# Count Not Cached
not_cached_count = df['decision'].value_counts().get('Not Cached', 0)

print(f'{not_cached_count} / {len(df)} requests are not cached, or  {not_cached_count / len(df) * 100:.2f}% percent')

# Count unique full_addr values
df['modified_full_addr'] = (df['full_addr'] // 64) * 64

unique_full_addrs_count = df['modified_full_addr'].nunique()

print(f'There are {unique_full_addrs_count} unique modified full_addr values')

# Count unique PCs
unique_ip_count = df['ip'].nunique()

print(f'There are {unique_ip_count} unique PCs')


# import matplotlib.pyplot as plt

# # Example: downsample the data by plotting every nth data point
# downsample_factor = 1000  # Plot every 1000th data point
# downsampled_accesses = df['modified_full_addr'][::downsample_factor]

# time_downsampled = list(range(0, len(df['modified_full_addr']), downsample_factor))

# # Plot downsampled data
# plt.figure(figsize=(10, 6))
# plt.scatter(time_downsampled, downsampled_accesses, color='blue', s=1, alpha=0.1)  # Adjust alpha for transparency
# # plt.plot(time_downsampled, downsampled_accesses, marker='o', linestyle='-', color='b')
# plt.xlabel('Access Order (Time)')
# plt.savefig('data/accesses.png')