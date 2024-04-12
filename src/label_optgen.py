from collections import defaultdict, deque
import csv

OPTGEN_VECTOR_SIZE = 128

class ADDR_INFO:
    def __init__(self):
        self.addr = 0
        self.last_quanta = 0
        self.PC = 0
        self.prefetched = False
        self.lru = 0

    def init(self, curr_quanta):
        self.last_quanta = curr_quanta
        self.PC = 0
        self.prefetched = False
        self.lru = 0

    def update(self, curr_quanta, _pc):
        self.last_quanta = curr_quanta
        self.PC = _pc

    def mark_prefetch(self):
        self.prefetched = True


class OPTgen:
    def __init__(self, size):
        self.num_cache = 0
        self.num_dont_cache = 0
        self.access = 0
        self.CACHE_SIZE = size
        self.liveness_history = deque(
            [0] * OPTGEN_VECTOR_SIZE, maxlen=OPTGEN_VECTOR_SIZE
        )
        self.addr_info = defaultdict(ADDR_INFO)

    def add_access(self, curr_quanta):
        self.access += 1
        self.liveness_history[curr_quanta] = 0

    def add_prefetch(self, curr_quanta):
        self.liveness_history[curr_quanta] = 0

    def should_cache(self, curr_quanta, last_quanta):
        is_cache = True
        i = last_quanta
        while i != curr_quanta:
            if self.liveness_history[i] >= self.CACHE_SIZE:
                is_cache = False
                break
            i = (i + 1) % len(self.liveness_history)

        if is_cache:
            i = last_quanta
            while i != curr_quanta:
                self.liveness_history[i] += 1
                i = (i + 1) % len(self.liveness_history)

        if is_cache:
            self.num_cache += 1
        else:
            self.num_dont_cache += 1

        return is_cache

    def get_num_opt_hits(self):
        return self.num_cache
    
def replace_addr_history_element(addr_history):
    lru_addr = None

    # Iterate over the addr_history dictionary for the given sampler_set
    for addr, info in addr_history.items():
        # Check if the lru attribute equals SAMPLER_WAYS - 1
        if info.lru == SAMPLER_WAYS - 1:
            lru_addr = addr
            break

    # If an address with the lru condition found, erase it from the dictionary
    if lru_addr is not None:
        del addr_history[sampler_set][lru_addr]


def label_optgen(cache_data_path, output_csv_path, optgen: OPTgen):
    cache_accesses = defaultdict(deque)
    prefetches = defaultdict(deque) 
    addr_history = {}
    curr_quanta = 0

    with open(cache_data_path, mode="r") as infile, open(
        output_csv_path, mode="w", newline=""
    ) as outfile:
        reader = csv.DictReader(infile)
        fieldnames = reader.fieldnames + ["IS_CACHED"]
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()

        for row in reader:
            address = int(row["full_addr"])
            pc = int(row["ip"])
            is_prefetch = row["type"] == 2

            # Simulate prefetch or access addition based on the type
            if is_prefetch:
                optgen.add_prefetch(curr_quanta)
            else:
                optgen.add_access(curr_quanta)

            # Use OPTgen to decide whether to cache this access
            # Assuming 'LastQuanta' is provided for simplicity; it would need to be calculated
            last_quanta = int(row["LastQuanta"])
            is_cached = optgen.should_cache(curr_quanta, last_quanta)

            # Write decision to CSV
            row["IS_CACHED"] = "IS_CACHED" if is_cached else "NOT_CACHED"
            writer.writerow(row)

            curr_quanta += 1  # Increment current quanta for next access

    return labels
