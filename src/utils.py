import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c", "--cache_data_path", type=str, default="data/labeled_cache_accesses.csv"
    )
    parser.add_argument(
        "-p", "--prefetch_data_path", type=str, default="data/prefetches.csv"
    )
    parser.add_argument("-l", "--learning_rate", type=float, default=5e-3)
    parser.add_argument("-d", "--hidden_dim", type=int, default=12)
    parser.add_argument("-w", "--ip_history_window", type=int, default=5)
    parser.add_argument("-e", "--num_epochs", type=int, default=10)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--eval_data", type=str, default="data/cache_accesses.csv")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--model_name", type=str, default="cache_repl_bce.pth")

    args = parser.parse_args()
    return args
