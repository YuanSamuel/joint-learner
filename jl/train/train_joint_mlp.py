import time
import torch

from torch import nn
from torch.optim.lr_scheduler import ExponentialLR
from jl.models.mlp_replacement import (
    CacheReplacementNN,
    CacheReplacementNNTransformer,
    CacheReplacementNNJointTransformer,
)
from jl.dataloaders.joint_dataloader import get_joint_dataloader
from jl.utils import parse_args, load_config
from jl.models.contrastive_encoder import ContrastiveEncoder
from jl.data_engineering.count_labels import count_labels
import jl.dataloaders.dataloader as dl


def train(args):
    print(f"------------------------------")
    config = load_config(args.config)
    print(f"Config: dim {args.hidden_dim} window {args.ip_history_window}")
    print("Init Dataloader")

    dataloader, valid_dataloader, _, num_pcs, num_pages = get_joint_dataloader(
        args.cache_data_path,
        args.ip_history_window,
        args.prefetch_data_path,
        config,
        args.batch_size,
        name=args.dataset,
    )

    # pos_count, neg_count = count_labels(dataloader)
    print(f"Num Prefetch PCs: {num_pcs}, Num Pages: {num_pages}")

    if args.use_transformer:
        feature_sizes = [len(dl.CACHE_IP_TO_IDX) + 1, num_pcs + 1, num_pages + 1, 65]
        model = CacheReplacementNNJointTransformer(
            num_features=feature_sizes, hidden_dim=args.hidden_dim
        )
    else:
        model = CacheReplacementNN(
            num_features=args.ip_history_window + config.sequence_length * 3 + 1,
            hidden_dim=args.hidden_dim,
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    best_model = model
    print(f"Using device: {device}")

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = ExponentialLR(optimizer, gamma=0.95)

    print("Begin Training")

    # Training loop
    num_epochs = args.num_epochs
    best_loss = float("inf")
    for epoch in range(num_epochs):
        # Train Phase
        model.train()
        start_time = time.time()
        total_loss = 0
        total_correct = 0
        train_zeroes = 0
        for batch, data in enumerate(dataloader):
            cache_features, prefetch_pcs, prefetch_pages, prefetch_offsets, labels = (
                data
            )
            cache_features, prefetch_pcs, prefetch_pages, prefetch_offsets, labels = (
                cache_features.to(device),
                prefetch_pcs.to(device),
                prefetch_pages.to(device),
                prefetch_offsets.to(device),
                labels.to(device),
            )
            optimizer.zero_grad()
            outputs = model(
                cache_features, prefetch_pcs, prefetch_pages, prefetch_offsets
            )
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            total_correct += count_correct(outputs, labels)
            train_zeroes += outputs[outputs < 0.5].shape[0]

            if batch % 1000 == 0 and batch != 0:
                ms_per_batch = (time.time() - start_time) * 1000 / batch
                acc = total_correct / (batch * args.batch_size) * 100
                print(
                    f"epoch {epoch+1} | batch {batch}/{len(dataloader)} batches"
                    + f" | ms/batch {ms_per_batch} | loss {total_loss:.4f}"
                    + f" | acc {acc:.4f}"
                )
        scheduler.step()

        print(
            f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss:.4f}, Zeroes: {train_zeroes}"
        )
        print(f"------------------------------")

        # Validation phase
        model.eval()
        valid_loss = 0
        valid_correct = 0
        valid_zeroes = 0
        with torch.no_grad():
            for batch, data in enumerate(valid_dataloader):
                (
                    cache_features,
                    prefetch_pcs,
                    prefetch_pages,
                    prefetch_offsets,
                    labels,
                ) = data

                (
                    cache_features,
                    prefetch_pcs,
                    prefetch_pages,
                    prefetch_offsets,
                    labels,
                ) = (
                    cache_features.to(device),
                    prefetch_pcs.to(device),
                    prefetch_pages.to(device),
                    prefetch_offsets.to(device),
                    labels.to(device),
                )

                outputs = model(
                    cache_features, prefetch_pcs, prefetch_pages, prefetch_offsets
                )
                loss = criterion(outputs, labels)
                valid_loss += loss.item()
                valid_correct += count_correct(outputs, labels)
                valid_zeroes += outputs[outputs < 0.5].shape[0]

        valid_loss /= len(valid_dataloader)
        valid_accuracy = valid_correct / len(valid_dataloader.dataset) * 100

        print(
            f"Validation Loss: {valid_loss:.4f}, Validation Accuracy: {valid_accuracy:.2f}%, Zeroes: {valid_zeroes}"
        )
        print(f"------------------------------")

        if valid_loss < best_loss:
            best_loss = valid_loss
            torch.save(model.state_dict(), f"./data/model/{args.model_name}.pth")
            best_model = model
        else:
            return best_model

    return best_model


def count_correct(outputs, labels):
    return (outputs > 0.5).float().eq(labels).sum().item()


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
