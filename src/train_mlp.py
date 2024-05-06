import time
import torch

from torch import nn
from torch.optim.lr_scheduler import ExponentialLR
from models.mlp_replacement import CacheReplacementNN
from dataloader import get_cache_dataloader
from utils import parse_args


def train(args):
    print(f"------------------------------")
    print(f"Config: dim {args.hidden_dim} window {args.ip_history_window}")
    print("Init Dataloader")
    dataloader = get_cache_dataloader(
        args.cache_data_path, args.ip_history_window, args.batch_size
    )

    model = CacheReplacementNN(
        num_features=args.ip_history_window + 1, hidden_dim=args.hidden_dim
    )

    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = ExponentialLR(optimizer, gamma=0.95)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    best_model = model
    print(f"Using device: {device}")

    print("Begin Training")
    model.train()

    # Training loop
    num_epochs = args.num_epochs
    best_loss = float("inf")
    for epoch in range(num_epochs):
        start_time = time.time()
        total_loss = 0
        total_correct = 0
        for batch, data in enumerate(dataloader):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            total_correct += count_correct(outputs, labels)

            if batch % 1000 == 0 and batch != 0:
                ms_per_batch = (time.time() - start_time) * 1000 / batch
                acc = total_correct / (1000 * args.batch_size)
                print(
                    f"epoch {epoch+1} | batch {batch}/{len(dataloader)} batches"
                    + f" | ms/batch {ms_per_batch} | loss {total_loss:.4f}"
                    + f" | acc {acc:.4f}"
                )
        scheduler.step()

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss:.4f}")
        print(f"------------------------------")

        if total_loss < best_loss:
            best_loss = total_loss
            torch.save(model.state_dict(), f"./data/model/{args.model_name}.pth")
            best_model = model
        else:
            return best_model

    return best_model


def count_correct(outputs, labels):
    return (outputs == labels).sum().item()


def trace_model(model, args):
    model.eval()
    model = model.to("cpu")
    example_input = torch.randint(
        0, 1 << 12, (args.ip_history_window + 1,), dtype=torch.float32
    )
    traced_model = torch.jit.trace(model, example_input)
    traced_model.save(f"./data/model/{args.model_name}_traced.pt")


if __name__ == "__main__":
    args = parse_args()
    model = train(args)

    trace_model(model, args)
