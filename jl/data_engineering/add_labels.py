import argparse
import csv
from jl.data_engineering.label_min_optimized import get_beladys

def process_csv(input_file, output_file, cache_size):
    print("Begin processing file")
    with open(input_file, mode="r") as file:
        csv_reader = csv.DictReader(file)
        rows = list(csv_reader)
        accesses = [(int(row["full_addr"]) >> 6 << 6) for row in rows]
        accesses_set = set(accesses)
        print(f"Unique accesses: {len(accesses_set)}")
        decisions = get_beladys(accesses, cache_size)

    print("Begin writing to output file")
    with open(output_file, mode="w", newline="") as file:
        fieldnames = csv_reader.fieldnames + [
            "decision"
        ]  # Append 'decision' to existing fieldnames
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        for row, decision in zip(rows, decisions):
            if decision == "In Cache":
                continue
            row["decision"] = decision
            writer.writerow(row)
    
    print("Finished writing to output file")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i", "--input", type=str, default="data/cache_accesses_sphinx.csv"
    )
    parser.add_argument(
        "-o", "--output", type=str, default="data/labeled_cache_sphinx.csv"
    )
    parser.add_argument("-c", "--cache_size", type=int, default=2048 * 2)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    input_csv_path = args.input
    output_csv_path = args.output
    cache_size = args.cache_size
    process_csv(input_csv_path, output_csv_path, cache_size)

