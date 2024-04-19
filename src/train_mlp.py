import time
import torch
from torch import nn

from models.mlp_replacement import CacheReplacementNN
from dataloader import get_cache_dataloader
from utils import parse_args

args = parse_args()

print("Init Dataloader")
dataloader = get_cache_dataloader(args.cache_data_path, args.ip_history_window, args.batch_size)

model = CacheReplacementNN(num_features=args.ip_history_window + 1, hidden_dim=args.hidden_dim)

criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate) 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
print(f"Using device: {device}")

print("Begin Training")
model.train()
start_time = time.time()

# Training loop
num_epochs = 20
best_loss = float("inf")
for epoch in range(num_epochs):
    total_loss = 0
    for batch, data in enumerate(dataloader):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        # print(inputs, outputs, labels)
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        if batch % 1000 == 0 and batch != 0:
            ms_per_batch = (time.time() - start_time) * 1000 / batch
            print(f'epoch {epoch+1} | batch {batch}/{len(dataloader)} batches | ms/batch {ms_per_batch} | loss {total_loss:.4f}')

    
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss:.4f}')
    print(f'------------------------------')

    if loss < best_loss:
        best_loss = loss
        torch.save(model.state_dict(), "./data/model/cache_repl_bce.pth")
