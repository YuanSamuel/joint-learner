import os
import requests
from urllib.parse import urlsplit
import sys

# Base URL with placeholder for dataset name
BASE_URL = 'https://dpc3.compas.cs.stonybrook.edu/champsim-traces/speccpu/{dataset_name}.champsimtrace.xz'

# Predefined list of dataset names (you can modify this list)
PREDEFINED_DATASET_NAMES = [
    '605.mcf_s-782B',
    '620.omnetpp_s-874B',
    '621.wrf_s-6673B',
    '401.bzip2-226B',
    '436.cactusADM-1804B',
    '470.lbm-1274B',
    '437.leslie3d-232B',
    '623.xalancbmk_s-165B'
]

# Function to download a file from a generated URL if it doesn't already exist
def download_file(dataset_name, folder="data/traces"):
    # Generate the full URL by replacing the placeholder with the actual dataset name
    url = BASE_URL.format(dataset_name=dataset_name)

    # Extract the file name from the URL
    file_name = os.path.basename(urlsplit(url).path)
    
    # Ensure the folder exists
    if not os.path.exists(folder):
        os.makedirs(folder)
    
    # Full path for the file
    file_path = os.path.join(folder, file_name)
    
    # Check if the file already exists
    if os.path.exists(file_path):
        print(f"File '{file_name}' already exists, skipping download.")
        return file_path
    
    # Download the file
    try:
        print(f"Downloading '{file_name}' from {url}...")
        response = requests.get(url, stream=True)
        response.raise_for_status()  # Raise an error for bad status codes
        with open(file_path, 'wb') as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)
        print(f"Downloaded '{file_name}' successfully!")
        return file_path
    except requests.exceptions.RequestException as e:
        print(f"Failed to download '{file_name}'. Error: {e}")
        return None

# Main function to handle downloading from the provided dataset name or from the predefined list
def main(dataset_names=None):
    if dataset_names:
        # Download files for the provided dataset names
        for dataset_name in dataset_names:
            download_file(dataset_name)
    else:
        # Download files for the predefined dataset names
        for dataset_name in PREDEFINED_DATASET_NAMES:
            download_file(dataset_name)

# Entry point for the script
if __name__ == "__main__":
    # Check if dataset names are passed via command line arguments
    if len(sys.argv) > 1:
        input_dataset_names = sys.argv[1:]  # Get all dataset names passed as arguments
        main(input_dataset_names)
    else:
        # If no dataset name is provided, use the predefined dataset names
        main()
