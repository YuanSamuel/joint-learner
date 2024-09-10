import csv
import lzma
import torch
from collections import deque
from torch.utils.data import Dataset, DataLoader
from data_engineering.benchmark import BenchmarkTrace

CACHE_IP_TO_IDX = {}
def get_cache_ip_idx(ip):
    if ip not in CACHE_IP_TO_IDX:
        CACHE_IP_TO_IDX[ip] = len(CACHE_IP_TO_IDX)
    return CACHE_IP_TO_IDX[ip]


def get_cache_data(cache_data_path, ip_history_window):
    with open(cache_data_path, mode="r") as file:
        csv_reader = csv.DictReader(file)
        data = []
        history_ips = deque()

        for row in csv_reader:
            ip_idx = get_cache_ip_idx(int(row["ip"]))

            current_recent_ips = [x for x in history_ips]

            while len(current_recent_ips) < ip_history_window:
                current_recent_ips.append(-1)

            data.append((ip_idx, current_recent_ips[-ip_history_window:], row["decision"]))

            if len(history_ips) >= ip_history_window * 2:
                history_ips.popleft()
            history_ips.append(ip_idx)

        return data


class CacheAccessDataset(Dataset):
    def __init__(self, data, ip_history_window, start, end):
        self.window = ip_history_window

        self.data = data[start:end]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        label = 1 if self.data[idx][2] == "Cached" else 0
        # ips_tensor = torch.tensor(self.get_n_most_recent_ips(idx, self.window), dtype=torch.int)
        return self.data[idx][0], self.data[idx][1], label


def cache_collate_fn(batch):
    ips, ip_histories, labels = zip(*batch)
    combined_features = [
        torch.tensor([s] + l, dtype=torch.float32) for s, l in zip(ips, ip_histories)
    ]
    features_tensor = torch.stack(combined_features, dim=0)
    labels_tensor = torch.tensor(labels, dtype=torch.float32).unsqueeze(1)

    return features_tensor, labels_tensor


def get_cache_dataloader(
    cache_data_path, ip_history_window, batch_size, train_pct=0.6, valid_pct=0.2
):
    data = get_cache_data(cache_data_path, ip_history_window)

    valid_start = int(len(data) * train_pct)
    eval_start = int(len(data) * (train_pct + valid_pct))

    train_dataset = CacheAccessDataset(data, ip_history_window, 0, valid_start)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=cache_collate_fn,
        num_workers=4,
        pin_memory=True if torch.cuda.is_available() else False,
    )

    valid_dataset = CacheAccessDataset(data, ip_history_window, valid_start, eval_start)
    valid_dataloader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=cache_collate_fn,
        num_workers=4,
    )

    eval_dataset = CacheAccessDataset(data, ip_history_window, eval_start, len(data))
    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=cache_collate_fn,
        num_workers=4,
    )

    return train_dataloader, valid_dataloader, eval_dataloader


def read_benchmark_trace(benchmark_path, config, args):
    """
    Reads and processes the trace for a benchmark
    """
    benchmark = BenchmarkTrace(config, args)

    if benchmark_path.endswith(".txt.xz"):
        with lzma.open(benchmark_path, mode="rt", encoding="utf-8") as f:
            benchmark.read_and_process_file(f)
    else:
        with open(benchmark_path, "r") as f:
            benchmark.read_and_process_file(f)

    return benchmark


class PrefetchInfo:
    def __init__(self, config):
        self.config = config
        self.offset_mask = (1 << config.offset_bits) - 1
        self.pc_mapping = {"oov": 0}
        self.page_mapping = {"oov": 0}
        self.pc_addrs = {}
        self.pc_addrs_idx = {}
        self.pc_data = [[]]
        self.count = {}
        self.cache_lines = {}
        self.cache_lines_idx = {}
        self.orig_addr = [0]
        self.data = [[0, 0, 0, 0, 0, 0]]


