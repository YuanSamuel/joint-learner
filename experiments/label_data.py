import os
import subprocess
import argparse

# Function to construct and run the command
def run_script(dataset_name, cache_size):
    input_file = f"data/collector_output/cache_accesses_{dataset_name}.csv"
    output_file = f"data/labeled_data/{dataset_name}_cs_{cache_size}_labeled.csv"

    if os.path.exists(output_file):
        print(f"Skipping {dataset_name} with cache size {cache_size} as it already exists.")
        return

    command = [
        "python", "-m", "jl.data_engineering.add_labels",
        "--input", input_file,
        "--output", output_file,
        "--cache_size", str(cache_size)
    ]
    print(f"Executing: {' '.join(command)}")
    subprocess.run(command, check=True)

# Function to process all datasets in the folder
def process_all_datasets(folder, cache_sizes):
    for file_name in os.listdir(folder):
        if file_name.startswith("cache_accesses_") and file_name.endswith(".csv"):
            dataset_name = file_name[len("cache_accesses_"):-len(".csv")]
            for cache_size in cache_sizes:
                run_script(dataset_name, cache_size)

# Function to process a specific dataset
def process_single_dataset(dataset_name, cache_sizes):
    for cache_size in cache_sizes:
        run_script(dataset_name, cache_size)

# Main function to handle command-line arguments
def main():
    parser = argparse.ArgumentParser(description="Run labeling script for dataset files.")
    parser.add_argument("--dataset_name", type=str, help="Specific dataset name to process.")
    parser.add_argument("--folder", type=str, default="data/collector_output", help="Folder containing dataset files.")
    args = parser.parse_args()

    # Cache sizes
    cache_sizes = [4096, 32768]

    # Process specific dataset if provided, otherwise process all datasets
    if args.dataset_name:
        process_single_dataset(args.dataset_name, cache_sizes)
    else:
        process_all_datasets(args.folder, cache_sizes)

if __name__ == "__main__":
    main()
