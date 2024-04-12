import torch
from torch.utils.data import Dataset, DataLoader

class CacheAccessDataset(Dataset):
    def __init__(self, cache_data_path):
        # Simulates a dataset of cache access patterns
        self.data = torch.randn(size, num_features)
        # Simulates binary labels: 1 for cache, 0 for not cache
        self.labels = torch.randint(0, 2, (size, 1), dtype=torch.float32)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]
    
def get_cache_dataloader(cache_data_path, batch_size):
    dataset = CacheAccessDataset(cache_data_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader