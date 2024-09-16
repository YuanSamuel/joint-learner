import csv
import torch
from collections import deque
from dataloader import PrefetchInfo, get_cache_ip_idx
from torch.utils.data import Dataset, DataLoader
from utils import has_dataset, save_dataset, load_dataset, split_dataset


class JointData:
    def __init__(
        self,
        cache_data_path,
        ip_history_window,
        prefetch_data_path,
        config,
    ):
        self.window = ip_history_window
        self.config = config
        self.prefetch_info = PrefetchInfo(config)

        print("Processing Cache Data")
        self.process_cache_data(cache_data_path)
        print("Processing Prefetch Data")
        self.process_prefetch_data(prefetch_data_path)
        print("Concat Pairs")
        self.make_pairs()

    def process_cache_data(self, cache_data_path):
        with open(cache_data_path, mode="r") as file:
            csv_reader = csv.DictReader(file)
            self.cache_data = []
            self.cache_timestamps = {}
            history_ips = deque()

            for row in csv_reader:
                ip_idx = get_cache_ip_idx(int(row["ip"]))
                current_recent_ips = [x for x in history_ips]

                while len(current_recent_ips) < self.window:
                    current_recent_ips.append(-1)

                self.cache_timestamps[int(row["timestamp"])] = len(self.cache_data)

                self.cache_data.append(
                    (ip_idx, current_recent_ips[-self.window :], row["decision"])
                )

                if len(history_ips) >= self.window * 2:
                    history_ips.popleft()

                history_ips.append(ip_idx)

    def process_prefetch_data(self, prefetch_data_path):
        with open(prefetch_data_path, mode="r") as file:
            self.prefetch_timestamps = {}
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
                self.prefetch_timestamps[int(row["timestamp"])] = len(
                    self.prefetch_info.data
                )
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
        cache_timestamps_list = list(self.cache_timestamps.keys())
        for idx, cache_timestamp in enumerate(cache_timestamps_list):
            if idx == 0:
                continue
            timestamp = cache_timestamp
            for i in range(timestamp, timestamp - 20, -1):
                if i in self.prefetch_timestamps:
                    pos_prefetch_idx = self.prefetch_timestamps[i]
                    self.data.append(
                        (self.cache_timestamps[timestamp], pos_prefetch_idx)
                    )
                    break


class JointDataset(Dataset):
    def __init__(
        self,
        joint_data,
    ):
        super().__init__()
        for key, value in vars(joint_data).items():
            setattr(self, key, value)

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

        return torch.cat(hists, dim=-1).float()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        prefetch_item = self.get_prefetch_item(self.data[idx][1])
        cache_item = self.cache_data[self.data[idx][0]]

        label = 1 if cache_item[2] == "Cached" else 0

        return prefetch_item, cache_item[0:2], label


def joint_collate_fn(batch):
    # Unzip the batch into separate lists for prefetch items and cache items
    prefetch_items, cache_items, labels = zip(*batch)

    ips, ip_histories = zip(*cache_items)
    combined_features = [
        torch.tensor([s] + l, dtype=torch.float32) for s, l in zip(ips, ip_histories)
    ]
    cache_features_tensor = torch.stack(combined_features, dim=0)

    labels_tensor = torch.tensor(labels, dtype=torch.float32).unsqueeze(1)
    prefetch_tensor = torch.stack(prefetch_items, dim=0)

    combined_tensor = torch.cat((prefetch_tensor, cache_features_tensor), dim=1)

    # Return a tuple of all the processed items
    return cache_features_tensor, labels_tensor


def get_joint_dataloader(
    cache_data_path,
    ip_history_window,
    prefetch_data_path,
    config,
    batch_size,
    train_pct=0.6,
    valid_pct=0.2,
    name=None,
):
    if name is not None and has_dataset(name):
        dataset = load_dataset(name)

    else:
        joint_data = JointData(
            cache_data_path, ip_history_window, prefetch_data_path, config
        )
        dataset = JointDataset(joint_data)

        if name is not None:
            save_dataset(name, dataset)

    train_dataset, valid_dataset, eval_dataset = split_dataset(
        dataset, train_pct, valid_pct
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=joint_collate_fn,
        num_workers=8,
        pin_memory=True if torch.cuda.is_available() else False,
    )

    valid_dataloader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=joint_collate_fn,
        num_workers=8,
        pin_memory=True if torch.cuda.is_available() else False,
    )

    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=joint_collate_fn,
        num_workers=8,
    )

    return (
        train_dataloader,
        valid_dataloader,
        eval_dataloader,
        len(joint_data.prefetch_info.pc_mapping),
        len(joint_data.prefetch_info.page_mapping),
    )
