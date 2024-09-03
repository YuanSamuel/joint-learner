import pandas as pd

# Replace 'your_file.csv' with the path to your CSV file
csv_file = 'data/xalan_large_labeled.csv'

# Read the CSV file into a DataFrame
df = pd.read_csv(csv_file)

# Count the number of occurrences of "Not Cached" in the "decision" column
not_cached_count = df['decision'].value_counts().get('Not Cached', 0)

print(f'{not_cached_count} / {len(df)} requests are not cached')
