import time
import torch

from torch import nn
from torch.optim.lr_scheduler import ExponentialLR
from models.contrastive_encoder import ContrastiveEncoder
from models.voyager import VoyagerEncoder
from dataloader import get_contrastive_dataloader
from utils import parse_args, load_config
from loss_fns.contrastive import ContrastiveLoss


def train(args):
    print(f"------------------------------")
    config = load_config(args.config)

    print("Init Dataloader")
    dataloader, num_pcs, num_pages = get_contrastive_dataloader(
        args.cache_data_path,
        args.ip_history_window,
        args.prefetch_data_path,
        config,
        args.batch_size,
        0,
        0.3
    )

    voyager_encoder = VoyagerEncoder(config, num_pcs, num_pages)
    cache_encoder = ContrastiveEncoder(
        args.ip_history_window + 1, config.contrastive_hidden_dim, config.contrastive_size
    )

    criterion = ContrastiveLoss()
    voyager_optimizer = torch.optim.Adam(voyager_encoder.parameters(), lr=args.learning_rate)
    cache_optimizer = torch.optim.Adam(cache_encoder.parameters(), lr=args.learning_rate)
    # scheduler = ExponentialLR(optimizer, gamma=0.95)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    voyager_encoder = voyager_encoder.to(device)
    cache_encoder = cache_encoder.to(device)
    best_voyager = voyager_encoder
    best_cache = cache_encoder
    print(f"Using device: {device}")

    print("Begin Training")
    voyager_encoder.train()
    cache_encoder.train()

    # Training loop
    num_epochs = args.num_epochs
    best_loss = float("inf")
    for epoch in range(num_epochs):
        start_time = time.time()
        total_loss = 0
        for batch, data in enumerate(dataloader):
            prefetch_input, cache_input, labels = data
            # print(prefetch_input)
            prefetch_input, cache_input, labels = (
                prefetch_input.to(device),
                cache_input.to(device),
                labels.to(device),
            )

            voyager_optimizer.zero_grad()
            cache_optimizer.zero_grad()

            prefetch_output = voyager_encoder(prefetch_input)
            cache_output = cache_encoder(cache_input)

            loss = criterion(prefetch_output, cache_output, labels)

            loss.backward()
            voyager_optimizer.step()
            cache_optimizer.step()

            total_loss += loss.item()

            if batch % 1000 == 0 and batch != 0:
                ms_per_batch = (time.time() - start_time) * 1000 / batch
                print(
                    f"epoch {epoch+1} | batch {batch}/{len(dataloader)} batches"
                    + f" | ms/batch {ms_per_batch} | loss {total_loss:.4f} | most recent loss {loss:.4f}"
                )
        # scheduler.step()

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss:.4f}")
        print(f"------------------------------")

        if total_loss < best_loss:
            best_loss = total_loss
            torch.save(voyager_encoder.state_dict(), f"./data/model/{args.model_name}_voyager.pth")
            torch.save(cache_encoder.state_dict(), f"./data/model/{args.model_name}_cache.pth")
            best_voyager = voyager_encoder
            best_cache = cache_encoder
        else:
            return best_voyager, best_cache

    return best_voyager, best_cache


if __name__ == "__main__":
    args = parse_args()
    model = train(args)
