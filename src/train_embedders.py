import time
import torch

from torch import nn
from torch.optim.lr_scheduler import ExponentialLR
from models.contrastive_encoder import ContrastiveEncoder
from models.voyager import VoyagerEncoder
from models.transformer_encoder import TransformerEncoder, PrefetchTransformerEncoder
from contrastive_dataloader import get_contrastive_dataloader
from utils import parse_args, load_config, tqdm
from loss_fns.contrastive import ContrastiveLoss

import dataloader as dl


def train(args):
    print(f"------------------------------")
    config = load_config(args.config)

    print("Init Dataloader")
    dataloader, valid_dataloader, _, num_pcs, num_pages = get_contrastive_dataloader(
        args.cache_data_path,
        args.ip_history_window,
        args.prefetch_data_path,
        config,
        args.batch_size,
        name=args.dataset,
    )

    print(f"Num Prefetch PCs: {num_pcs}, Num Pages: {num_pages}")

    if args.use_transformer:
        voyager_encoder = PrefetchTransformerEncoder(
            [num_pcs + 1, num_pages + 1, 65], args.hidden_dim, args.hidden_dim
        )
        cache_encoder = TransformerEncoder(
            len(dl.CACHE_IP_TO_IDX) + 1, args.hidden_dim, args.hidden_dim
        )
    else:
        voyager_encoder = ContrastiveEncoder(
            config.sequence_length * 3, args.hidden_dim, args.hidden_dim
        )
        cache_encoder = ContrastiveEncoder(
            args.ip_history_window + 1, args.hidden_dim, args.hidden_dim
        )

    criterion = ContrastiveLoss()
    voyager_optimizer = torch.optim.Adam(
        voyager_encoder.parameters(), lr=args.learning_rate
    )
    cache_optimizer = torch.optim.Adam(
        cache_encoder.parameters(), lr=args.learning_rate
    )
    # scheduler = ExponentialLR(optimizer, gamma=0.95)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    voyager_encoder = voyager_encoder.to(device)
    cache_encoder = cache_encoder.to(device)
    best_voyager = voyager_encoder
    best_cache = cache_encoder
    print(f"Using device: {device}")

    print("Begin Training")

    num_epochs = args.num_epochs
    best_loss = float("inf")
    for epoch in range(num_epochs):
        # Training loop
        voyager_encoder.train()
        cache_encoder.train()

        start_time = time.time()
        total_loss = 0
        for batch, data in enumerate(tqdm(dataloader, desc=f"Epoch {epoch + 1}")):
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

            # tqdm.write(f"CACHE FEATURES: {cache_features}")
            # tqdm.write(f"PREFETCH PCS: {prefetch_pcs}")
            # tqdm.write(f"PREFETCH PAGES: {prefetch_pages}")
            # tqdm.write(f"PREFETCH OFFSETS: {prefetch_offsets}")
            # tqdm.write(f"LABELS: {labels}")

            voyager_optimizer.zero_grad()
            cache_optimizer.zero_grad()

            prefetch_output = voyager_encoder(
                prefetch_pcs, prefetch_pages, prefetch_offsets
            )
            cache_output = cache_encoder(cache_features)

            loss = criterion(prefetch_output, cache_output, labels)

            loss.backward()
            voyager_optimizer.step()
            cache_optimizer.step()

            total_loss += loss.item()

            if batch % 1000 == 0 and batch != 0:
                ms_per_batch = (time.time() - start_time) * 1000 / batch
                tqdm.write(
                    f"epoch {epoch+1} | batch {batch}/{len(dataloader)} batches"
                    + f" | ms/batch {ms_per_batch} | loss {total_loss:.4f} | avg loss {total_loss / batch:.4f}"
                )
        # scheduler.step()

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss:.4f}")
        print(f"------------------------------")

        # Validation loop
        voyager_encoder.eval()
        cache_encoder.eval()
        val_loss = 0
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
                prefetch_output = voyager_encoder(
                    prefetch_pcs, prefetch_pages, prefetch_offsets
                )
                cache_output = cache_encoder(cache_features)

                loss = criterion(prefetch_output, cache_output, labels)
                val_loss += loss.item()

        print(f"Epoch [{epoch+1}/{num_epochs}], Validation Loss: {val_loss:.4f}")
        print(f"------------------------------")

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(
                voyager_encoder.state_dict(),
                f"./data/model/{args.model_name}_voyager.pth",
            )
            torch.save(
                cache_encoder.state_dict(), f"./data/model/{args.model_name}_cache.pth"
            )
            best_voyager = voyager_encoder
            best_cache = cache_encoder
        else:
            return best_voyager, best_cache

    return best_voyager, best_cache


if __name__ == "__main__":
    args = parse_args()
    model = train(args)
