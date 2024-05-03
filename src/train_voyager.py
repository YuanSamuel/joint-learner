import time
import torch

from utils import parse_args, load_config
from dataloader import read_benchmark_trace
from models.voyager import Voyager
from loss_fns.hierarchical_ce import HierarchicalCrossEntropyWithLogitsLoss
from eval.measure_voyager import (
    count_page_correct,
    count_offset_correct,
    count_overall_correct,
)


def train(args):
    print(f"------------------------------")

    # Parse config file
    config = load_config(args.config)
    print(config)
    # print(f"Config: dim {args.hidden_dim} window {args.ip_history_window}")

    print("Init Dataloader")
    benchmark = read_benchmark_trace(args.prefetch_data_path, config, args)

    # Create and compile the model
    model = Voyager(config, benchmark.num_pcs(), benchmark.num_pages())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"Using device: {device}")

    dataloader = benchmark.split()

    num_offsets = 1 << config.offset_bits
    criterion = HierarchicalCrossEntropyWithLogitsLoss(
        multi_label=config.multi_label, num_offsets=num_offsets
    )
    # scheduler = ExponentialLR(optimizer, gamma=0.95)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    best_model = model

    print("Begin Training")
    model.train()

    # Training loop
    num_epochs = args.num_epochs
    best_loss = float("inf")
    for epoch in range(num_epochs):
        start_time = time.time()
        total_loss = 0
        total_page_correct = 0
        total_offset_correct = 0
        total_correct = 0
        for batch, data in enumerate(dataloader):
            _, _, x, y_page, y_offset = data
            x, y_page, y_offset = x.to(device), y_page.to(device), y_offset.to(device)
            optimizer.zero_grad()

            outputs = model(x)
            loss = criterion(outputs, (y_page, y_offset))

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            total_page_correct += count_page_correct(
                y_page, outputs, num_offsets, config
            )
            total_offset_correct += count_offset_correct(
                y_offset, outputs, num_offsets, config
            )
            total_correct += count_overall_correct(
                y_page, y_offset, outputs, num_offsets, config
            )

            if batch % 1000 == 0 and batch != 0:
                ms_per_batch = (time.time() - start_time) * 1000 / batch
                page_acc = total_page_correct / (batch * args.batch_size) * 100
                offset_acc = total_offset_correct / (batch * args.batch_size) * 100
                overall_acc = total_correct / (batch * args.batch_size) * 100
                print(
                    f"epoch {epoch+1} | batch {batch}/{len(dataloader)} batches"
                    + f" | ms/batch {ms_per_batch} | loss {total_loss:.4f}" 
                    + f" | page_acc {page_acc:.4f} | offset_acc {offset_acc:.4f}"
                    + f" | overall_acc {overall_acc:.4f}"
                )
        # scheduler.step()

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss:.4f}")
        print(f"------------------------------")

        if loss < best_loss:
            best_loss = loss
            torch.save(model.state_dict(), f"./data/model/{args.model_name}.pth")
            best_model = model
        else:
            return best_model

    return best_model


if __name__ == "__main__":
    args = parse_args()
    model = train(args)
