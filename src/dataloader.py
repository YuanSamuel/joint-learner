import csv
import lzma
import random
import torch
from collections import deque
from torch.utils.data import Dataset, DataLoader
from data_engineering.benchmark import BenchmarkTrace
from collections import namedtuple


def get_cache_data(cache_data_path, ip_history_window):
    with open(cache_data_path, mode="r") as file:
        csv_reader = csv.DictReader(file)
        data = []
        history_ips = deque()
        seen_ips = set()

        for row in csv_reader:
            ip = int(row["ip"])
            current_recent_ips = [x for x in history_ips if x != ip]

            while len(current_recent_ips) < ip_history_window:
                current_recent_ips.append(-1)

            data.append((ip, current_recent_ips[:ip_history_window], row["decision"]))

            if ip in seen_ips:
                history_ips = deque([x for x in history_ips if x != ip])
                seen_ips.remove(ip)

            if len(history_ips) >= ip_history_window * 2:
                removed_ip = history_ips.popleft()
                seen_ips.remove(removed_ip)

            history_ips.append(ip)
            seen_ips.add(ip)

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


def get_cache_dataloader(cache_data_path, ip_history_window, batch_size, train_pct=0.3, valid_pct=0.1):
    data = get_cache_data(cache_data_path, ip_history_window)

    valid_start = int(len(data) * train_pct)
    eval_start = int(len(data) * (train_pct + valid_pct))

    train_dataset = CacheAccessDataset(
        data, ip_history_window, 0, valid_start
    )
    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, collate_fn=cache_collate_fn
    )

    valid_dataset = CacheAccessDataset(
        data, ip_history_window, valid_start, eval_start
    )
    valid_dataloader = DataLoader(
        valid_dataset, batch_size=batch_size, shuffle=True, collate_fn=cache_collate_fn
    )

    eval_dataset = CacheAccessDataset(
        data, ip_history_window, eval_start, len(data)
    )
    eval_dataloader = DataLoader(
        eval_dataset, batch_size=batch_size, shuffle=True, collate_fn=cache_collate_fn
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


class ContrastiveDataset(Dataset):
    def __init__(
        self,
        cache_data_path,
        ip_history_window,
        prefetch_data_path,
        config,
        start_pct,
        end_pct,
    ):
        self.window = ip_history_window
        self.config = config
        self.prefetch_info = PrefetchInfo(config)

        self.process_cache_data(cache_data_path)
        self.process_prefetch_data(prefetch_data_path)
        self.make_pairs()
        self.data = self.data[
            int(len(self.data) * start_pct) : int(len(self.data) * end_pct)
        ]

    def process_cache_data(self, cache_data_path):
        with open(cache_data_path, mode="r") as file:
            csv_reader = csv.DictReader(file)
            self.cache_data = []
            self.cache_timestamps = {}
            history_ips = deque()
            seen_ips = set()

            for row in csv_reader:
                ip = int(row["ip"])
                current_recent_ips = [x for x in history_ips if x != ip]

                while len(current_recent_ips) < self.window:
                    current_recent_ips.append(-1)

                self.cache_timestamps[int(row["timestamp"])] = len(self.cache_data)
                self.cache_data.append(
                    (ip, current_recent_ips[: self.window], int(row["timestamp"]))
                )

                if ip in seen_ips:
                    history_ips = deque([x for x in history_ips if x != ip])
                    seen_ips.remove(ip)

                if len(history_ips) >= self.window * 2:
                    removed_ip = history_ips.popleft()
                    seen_ips.remove(removed_ip)

                history_ips.append(ip)
                seen_ips.add(ip)

    def process_prefetch_data(self, prefetch_data_path):
        with open(prefetch_data_path, mode="r") as file:
            csv_reader = csv.DictReader(file)
            for idx, row in enumerate(csv_reader):
                addr = int(row["addr"]) >> 6 << 6
                pc = int(row["ip"])
                cache_line = addr >> 6
                page, offset = (
                    cache_line >> self.config.offset_bits,
                    cache_line & self.prefetch_info.offset_mask,
                )

                if pc not in self.prefetch_info.pc_mapping:
                    self.prefetch_info.pc_mapping[pc] = len(
                        self.prefetch_info.pc_mapping
                    )
                    # These are needed for PC localization
                    self.prefetch_info.pc_addrs[self.prefetch_info.pc_mapping[pc]] = []
                    self.prefetch_info.pc_addrs_idx[
                        self.prefetch_info.pc_mapping[pc]
                    ] = 0
                    self.prefetch_info.pc_data.append(
                        [
                            0
                            for _ in range(
                                self.prefetch_info.config.sequence_length
                                + self.prefetch_info.config.prediction_depth
                            )
                        ]
                    )

                if page not in self.prefetch_info.page_mapping:
                    self.prefetch_info.page_mapping[page] = len(
                        self.prefetch_info.page_mapping
                    )

                # Needed for delta localization
                if (
                    self.prefetch_info.page_mapping[page],
                    offset,
                ) not in self.prefetch_info.count:
                    self.prefetch_info.count[
                        (self.prefetch_info.page_mapping[page], offset)
                    ] = 0
                self.prefetch_info.count[
                    (self.prefetch_info.page_mapping[page], offset)
                ] += 1

                self.prefetch_info.pc_addrs[self.prefetch_info.pc_mapping[pc]].append(
                    (self.prefetch_info.page_mapping[page], offset)
                )

                # Needed for spatial localization
                if cache_line not in self.prefetch_info.cache_lines:
                    self.prefetch_info.cache_lines[cache_line] = []
                    self.prefetch_info.cache_lines_idx[cache_line] = 0
                self.prefetch_info.cache_lines[cache_line].append(idx)

                # Include the instruction ID for generating the prefetch file for running
                # in the ML-DPC modified version of ChampSim.
                # See github.com/Quangmire/ChampSim
                self.prefetch_info.data.append(
                    [
                        idx,
                        self.prefetch_info.pc_mapping[pc],
                        self.prefetch_info.page_mapping[page],
                        offset,
                        len(
                            self.prefetch_info.pc_data[
                                self.prefetch_info.pc_mapping[pc]
                            ]
                        ),
                        int(row["timestamp"]),
                    ]
                )
                self.prefetch_info.orig_addr.append(cache_line)
                self.prefetch_info.pc_data[self.prefetch_info.pc_mapping[pc]].append(
                    len(self.prefetch_info.data) - 1
                )
        self.prefetch_info.data = torch.as_tensor(self.prefetch_info.data)
        self.prefetch_info.pc_data = [
            torch.as_tensor(item) for item in self.prefetch_info.pc_data
        ]

    def make_pairs(self):
        self.data = []
        for idx, prefetch_item in enumerate(self.prefetch_info.data):
            if idx == 0:
                continue
            timestamp = prefetch_item[5]
            found_cache = False
            for i in range(timestamp - 20, timestamp + 20):
                if i in self.cache_timestamps:
                    pos_cache_idx = self.cache_timestamps[i]
                    found_cache = True
                    break

            if found_cache:
                neg_item = random.choice(self.cache_data)
                while abs(neg_item[2] - timestamp) < 20:
                    neg_item = random.choice(self.cache_data)
                self.data.append((idx, pos_cache_idx, neg_item))

    def get_prefetch_item(self, idx):
        hists = []
        cur_pc = self.prefetch_info.data[idx, 1].item()
        end = self.prefetch_info.data[idx, 4].item()
        start = end - self.config.sequence_length - self.config.prediction_depth

        if self.config.pc_localized:
            indices = self.prefetch_info.pc_data[cur_pc][start : end + 1].long()
            hist = self.prefetch_info.data[indices]
            page_hist = hist[: self.config.sequence_length, 2]
            offset_hist = hist[: self.config.sequence_length, 3]
            if self.config.use_current_pc:
                pc_hist = hist[1 : self.config.sequence_length + 1, 1]
            else:
                pc_hist = hist[: self.config.sequence_length, 1]
            hists.extend([pc_hist, page_hist, offset_hist])

        return torch.cat(hists, dim=-1)

    def __len__(self):
        return len(self.data) * 2

    def __getitem__(self, idx):
        if idx % 2 == 0:
            get_idx = idx // 2
            prefetch_item = self.get_prefetch_item(self.data[get_idx][0])
            cache_item = self.cache_data[self.data[get_idx][1]]
        else:
            get_idx = idx // 2
            prefetch_item = self.get_prefetch_item(self.data[get_idx][0])
            cache_item = self.data[get_idx][2]

        return prefetch_item, cache_item[0:2], 1 if idx % 2 == 0 else 0


def contrastive_collate_fn(batch):
    # Unzip the batch into separate lists for prefetch items and cache items
    prefetch_items, cache_items, labels = zip(*batch)

    ips, ip_histories = zip(*cache_items)
    combined_features = [
        torch.tensor([s] + l, dtype=torch.float32) for s, l in zip(ips, ip_histories)
    ]
    cache_features_tensor = torch.stack(combined_features, dim=0)

    labels_tensor = torch.tensor(labels, dtype=torch.float32)
    prefetch_tensor = torch.stack(prefetch_items, dim=0)

    # Return a tuple of all the processed items
    return prefetch_tensor, cache_features_tensor, labels_tensor


def get_contrastive_dataloader(
    cache_data_path,
    ip_history_window,
    prefetch_data_path,
    config,
    batch_size,
    start_pct=0,
    end_pct=0.3,
):
    dataset = ContrastiveDataset(
        cache_data_path,
        ip_history_window,
        prefetch_data_path,
        config,
        start_pct,
        end_pct,
    )
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, collate_fn=contrastive_collate_fn
    )
    return (
        dataloader,
        len(dataset.prefetch_info.pc_mapping),
        len(dataset.prefetch_info.page_mapping),
    )
