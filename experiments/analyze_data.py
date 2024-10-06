import os
import subprocess
import sys

# Path to the directory containing the dataset files
DATA_DIR = 'data/labeled_data/'

def get_dataset_name_from_file(file_name):
    # Removes "_labeled.csv" from the end of the file name to get the dataset name
    return file_name.replace("_labeled.csv", "")

def run_script_on_dataset(dataset_name):
    # Create the command to execute
    dataset_file = f"{DATA_DIR}{dataset_name}_labeled.csv"
    if os.path.exists(dataset_file):
        command = ["python", "-m", "jl.data_engineering.get_dataset_stats", dataset_file]
        try:
            subprocess.run(command, check=True)
            print(f"Executed script on {dataset_name}")
        except subprocess.CalledProcessError as e:
            print(f"Failed to execute script on {dataset_name}: {e}")
    else:
        print(f"Dataset file {dataset_file} does not exist.")

def run_on_all_datasets():
    # Get all files in the data/labeled_data/ directory
    for file_name in os.listdir(DATA_DIR):
        if file_name.endswith("_labeled.csv"):
            dataset_name = get_dataset_name_from_file(file_name)
            run_script_on_dataset(dataset_name)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        # If a dataset name is provided, run the script only for that dataset
        dataset_name = sys.argv[1]
        run_script_on_dataset(dataset_name)
    else:
        # Otherwise, run the script for all datasets in the folder
        run_on_all_datasets()
