import csv
import torch
from collections import deque
from torch.utils.data import Dataset, DataLoader

class CacheAccessDataset(Dataset):
    def __init__(self, cache_data_path, ip_history_window):
        self.window = ip_history_window

        with open(cache_data_path, mode="r") as file:
            csv_reader = csv.DictReader(file)
            self.data = []
            history_ips = deque()
            seen_ips = set()

            for row in csv_reader:
                ip = int(row['ip']) >> 6 << 6
                current_recent_ips = [x for x in history_ips if x != ip]

                while len(current_recent_ips) < self.window:
                    current_recent_ips.append(-1)

                self.data.append((ip, current_recent_ips[:self.window], row['decision']))

                if ip in seen_ips:
                    history_ips = deque([x for x in history_ips if x != ip])
                    seen_ips.remove(ip)

                if len(history_ips) >= self.window * 2:
                    removed_ip = history_ips.popleft()
                    seen_ips.remove(removed_ip)

                history_ips.append(ip)
                seen_ips.add(ip)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        label = 1 if self.data[idx][2] == 'Cached' else 0
        # ips_tensor = torch.tensor(self.get_n_most_recent_ips(idx, self.window), dtype=torch.int)
        return self.data[idx][0], self.data[idx][1], label

def cache_collate_fn(batch):
    ips, ip_histories, labels = zip(*batch)
    combined_features = [torch.tensor([s] + l, dtype=torch.float32) for s, l in zip(ips, ip_histories)]
    features_tensor = torch.stack(combined_features, dim=0)
    labels_tensor = torch.tensor(labels, dtype=torch.float32).unsqueeze(1)

    return features_tensor, labels_tensor


def get_cache_dataloader(cache_data_path, ip_history_window, batch_size):
    dataset = CacheAccessDataset(cache_data_path, ip_history_window)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=cache_collate_fn)
    return dataloader