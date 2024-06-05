import time
import torch
from torch import nn

from models.mlp_replacement import CacheReplacementNN
from dataloader import get_cache_dataloader
from utils import parse_args

args = parse_args()

print("Init Dataloader")
_, dataloader = get_cache_dataloader(args.cache_data_path, args.ip_history_window, args.batch_size)

model = CacheReplacementNN(num_features=args.ip_history_window + 1, hidden_dim=args.hidden_dim)

state_dict = torch.load(f"./data/model/{args.model_name}.pth")
model.load_state_dict(state_dict)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
print(f"Using device: {device}")

print("Begin Eval")
model.eval()
start_time = time.time()

correct = 0
total = 0
with torch.no_grad():
    for batch, data in enumerate(dataloader):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        outputs = model(inputs)
        predicted = (outputs > 0.5).float()

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        if batch % 10000 == 0 and batch != 0:
            ms_per_batch = (time.time() - start_time) * 1000 / batch
            print(f'batch {batch}/{len(dataloader)} | accuracy {correct}/{total} | ms/batch {ms_per_batch}')

print(f'Accuracy: {correct}/{total} = {100 * correct / total:.2f}%')
