import torch
from torch import nn

from models.mlp_replacement import CacheReplacementNN
from dataloader import get_cache_dataloader
from utils import parse_args

args = parse_args()

dataloader = get_cache_dataloader(args.cache_data_path, args.ip_history_window, args.batch_size)

model = CacheReplacementNN(num_features=args.ip_history_window + 1, hidden_dim=args.hidden_dim)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate) 

# Training loop
num_epochs = 20
best_loss = float("inf")
for epoch in range(num_epochs):
    total_loss = 0
    for inputs, labels in dataloader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss:.4f}')

    if loss < best_loss:
        best_loss = loss
        torch.save(model.state_dict(), "./data/model/cache_repl.pth")
