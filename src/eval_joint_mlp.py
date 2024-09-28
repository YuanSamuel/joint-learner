import time
import torch

from torch import nn
from torch.optim.lr_scheduler import ExponentialLR
from models.mlp_replacement import (
    CacheReplacementNN,
    CacheReplacementNNTransformer,
    CacheReplacementNNJointTransformer,
)
from joint_dataloader import get_joint_dataloader
from utils import parse_args, load_config
from models.contrastive_encoder import ContrastiveEncoder
from data_engineering.count_labels import count_labels
import dataloader as dl


def eval(args):
    print(f"------------------------------")
    config = load_config(args.config)
    print(f"Config: dim {args.hidden_dim} window {args.ip_history_window}")
    print("Init Dataloader")

    _, _, dataloader, num_pcs, num_pages = get_joint_dataloader(
        args.cache_data_path,
        args.ip_history_window,
        args.prefetch_data_path,
        config,
        args.batch_size,
        name=args.dataset,
    )

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

    state_dict = torch.load(f"./data/model/{args.model_name}.pth")
    model.load_state_dict(state_dict)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"Using device: {device}")

    model.eval()
    start_time = time.time()

    correct = 0
    zeroes = 0
    total = 0

    with torch.no_grad():
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

            outputs = model(
                cache_features, prefetch_pcs, prefetch_pages, prefetch_offsets
            )

            total += labels.size(0)
            correct += count_correct(outputs, labels)
            zeroes += outputs[outputs < 0.5].shape[0]

            if batch % 10000 == 0 and batch != 0:
                ms_per_batch = (time.time() - start_time) * 1000 / batch
                print(
                    f"batch {batch}/{len(dataloader)} | accuracy {correct}/{total} | ms/batch {ms_per_batch}"
                )

    accuracy = correct / len(dataloader.dataset) * 100

    print(f"Accuracy: {accuracy:.2f}%, Zeroes: {zeroes}")
    print(f"------------------------------")


def count_correct(outputs, labels):
    return (outputs > 0.5).float().eq(labels).sum().item()


if __name__ == "__main__":
    args = parse_args()
    eval(args)
