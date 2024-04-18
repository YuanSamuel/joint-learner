import csv
import torch
from torch.utils.data import Dataset, DataLoader

class CacheAccessDataset(Dataset):
    def __init__(self, cache_data_path, ip_history_window):
        self.window = ip_history_window
        with open(cache_data_path, mode="r") as file:
            csv_reader = csv.DictReader(file)
            self.data = []
            for row in csv_reader:
                self.data.append((int(row['ip']), row['decision']))
    
    def __len__(self):
        return len(self.data)
    
    def get_n_most_recent_ips(self, idx, n):
        ips = []
        current_ip = self.data[idx][0]
        for i in range(idx, -1 , -1):
            prev_ip = self.data[i][0]
            if not prev_ip in ips and prev_ip != current_ip:
                ips.append(prev_ip)
                if len(ips) == n:
                    break

        ips.extend([-1] * (n - len(ips)))
        return ips
    
    def __getitem__(self, idx):
        label = 1 if self.data[idx][1] == 'Cached' else 0
        # ips_tensor = torch.tensor(self.get_n_most_recent_ips(idx, self.window), dtype=torch.int)
        return self.data[idx][0], self.get_n_most_recent_ips(idx, self.window), label

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