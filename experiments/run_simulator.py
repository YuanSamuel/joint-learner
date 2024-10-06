import os
import subprocess
import sys

# Define paths and command parameters as constants
SIM_BIN_PATH = "sim/bin/champsim"
WARMUP_INSTRUCTIONS_SM = 50000000
SIMULATION_INSTRUCTIONS_SM = 50000000
WARMUP_INSTRUCTIONS_LG = 100000000
SIMULATION_INSTRUCTIONS_LG = 300000000
TRACES_FOLDER = "data/traces/"
OUTPUT_FOLDER = "data/collector_output/"

def run_simulation(dataset_file):
    """
    Executes the Champsim command for the given dataset file.
    """
    dataset_name = ".".join(dataset_file.split(".")[:-2])
    dataset_path = os.path.join(TRACES_FOLDER, dataset_file)  # Full path to the dataset file
    name = dataset_name  # Use the dataset name as the name parameter
    output_file_lg = os.path.join(OUTPUT_FOLDER, f"cache_accesses_{dataset_name}_lg.csv")
    output_file_sm = os.path.join(OUTPUT_FOLDER, f"cache_accesses_{dataset_name}_sm.csv")

    if not os.path.exists(output_file_lg):
        # Large configuration
        command_lg = [
            SIM_BIN_PATH,
            f"--warmup_instructions={WARMUP_INSTRUCTIONS_LG}",
            f"--simulation_instructions={SIMULATION_INSTRUCTIONS_LG}",
            f"--name={name}_lg",
            dataset_path
        ]

        print(f"Running command: {' '.join(command_lg)}")
        subprocess.run(command_lg)
    else:
        print(f"Skipping {dataset_name} large: Output file {output_file_lg} already exists.")

    if not os.path.exists(output_file_sm):
        # Small configuration
        command_sm = [
            SIM_BIN_PATH,
            f"--warmup_instructions={WARMUP_INSTRUCTIONS_SM}",
            f"--simulation_instructions={SIMULATION_INSTRUCTIONS_SM}",
            f"--name={name}_sm",
            dataset_path
        ]

        print(f"Running command: {' '.join(command_sm)}")
        subprocess.run(command_sm)
    else:
        print(f"Skipping {dataset_name} small: Output file {output_file_sm} already exists.")


def main():
    """
    Main function to check for a specified dataset or run for all datasets.
    """
    # Check if a specific dataset is passed as an argument
    if len(sys.argv) > 1:
        specified_dataset = sys.argv[1]
        dataset_file = specified_dataset + ".champsimtrace.xz"

        # Check if the specified dataset file exists
        if not os.path.exists(os.path.join(TRACES_FOLDER, dataset_file)):
            print(f"Error: Dataset file {dataset_file} does not exist in {TRACES_FOLDER}")
            sys.exit(1)

        # Run simulation for the specified dataset
        run_simulation(dataset_file)
    else:
        # Get all files in the data/ folder if no dataset is specified
        data_files = [f for f in os.listdir(TRACES_FOLDER) if f.endswith(".champsimtrace.xz")]

        # Loop through each dataset file and run the simulation
        for data_file in data_files:
            run_simulation(data_file)

    print("All simulations completed.")

if __name__ == "__main__":
    main()
