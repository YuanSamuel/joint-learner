import csv
from label_min_optimized import get_beladys


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
            row["decision"] = decision
            writer.writerow(row)


input_csv_path = "data/cache_accesses_xalancmk_s_20m.csv"
output_csv_path = "data/labeled_cache_xalancmk_s_20m.csv"
cache_size = 2048 * 2

process_csv(input_csv_path, output_csv_path, cache_size)
