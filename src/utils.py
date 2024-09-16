import argparse
import os
import torch
import yaml
from torch.utils.data import Subset


from types import SimpleNamespace


def load_config(config_path, debug=False):
    """
    Loads config file and applies any necessary modifications to it
    """
    # Parse config file
    with open(config_path, "r") as f:
        data = yaml.safe_load(f)
        config = SimpleNamespace(**data)

    if not config.global_stream and not config.pc_localized:
        print(
            'Invalid config file. Either or both "global_stream" and "pc_localized" must be true'
        )
        exit()

    if config.global_output and config.pc_localized:
        print(
            'Invalid config file. "global_output" and "pc_localized" cannot both be true'
        )
        exit()

    if not config.global_output and not config.pc_localized:
        print(
            'Invalid config file. "global_output" and "pc_localized" cannot both be false'
        )
        exit()

    # If the debug flag was raised, reduce the number of steps to have faster epochs
    if debug:
        config.steps_per_epoch = 16000

    return config


def save_dataset(name, dataset):
    """
    Saves the dataset to disk
    """
    torch.save(dataset, f"./data/datasets/{name}.pt")


def has_dataset(name):
    """
    Checks if the dataset exists on disk
    """
    return os.path.exists(f"./data/datasets/{name}.pt")


def load_dataset(name):
    """
    Loads the dataset from disk
    """
    return torch.load(f"./data/datasets/{name}.pt")


def split_dataset(dataset, train_pct=0.6, valid_pct=0.2):
    valid_start = int(len(dataset) * train_pct)
    eval_start = int(len(dataset) * (train_pct + valid_pct))

    train_indices = list(range(0, valid_start))
    valid_indices = list(range(valid_start, eval_start))
    eval_indices = list(range(eval_start, len(dataset)))

    train_dataset = Subset(dataset, train_indices)
    valid_dataset = Subset(dataset, valid_indices)
    eval_dataset = Subset(dataset, eval_indices)

    return train_dataset, valid_dataset, eval_dataset


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c", "--cache_data_path", type=str, default="data/labeled_cache_accesses.csv"
    )
    parser.add_argument(
        "-p", "--prefetch_data_path", type=str, default="data/prefetches.csv"
    )
    parser.add_argument("--config", type=str, default="configs/base_voyager.yaml")
    parser.add_argument("-l", "--learning_rate", type=float, default=0.0001)
    parser.add_argument("-d", "--hidden_dim", type=int, default=12)
    parser.add_argument("-w", "--ip_history_window", type=int, default=5)
    parser.add_argument("-e", "--num_epochs", type=int, default=10)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--eval_data", type=str, default="data/cache_accesses.csv")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--model_name", type=str, default="cache_repl_bce")
    parser.add_argument("--encoder_name", type=str, default="none")
    parser.add_argument("--dataset", type=str, default=None)

    args = parser.parse_args()
    return args
