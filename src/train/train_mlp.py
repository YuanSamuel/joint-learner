import torch
from torch import nn

from src.models import CacheReplacementNN
from src.dataloader import get_cache_dataloader
from src.utils import parse_args

args = parse_args()

dataloader = get_cache_dataloader(args.cache_data_path, batch_size=args.batch_size)

# Define the model
model = CacheReplacementNN(num_features=10)

# Define the loss function and optimizer
criterion = nn.BCELoss()  # Binary Cross-Entro  py Loss for binary classification
optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate) 

# Training loop
num_epochs = 20
for epoch in range(num_epochs):
    for inputs, labels in dataloader:
        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)
        
        # Calculate loss
        loss = criterion(outputs, labels)
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
    
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
