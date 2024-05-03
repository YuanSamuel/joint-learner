import time
import torch

from utils import parse_args, load_config
from dataloader import read_benchmark_trace
from models.voyager import Voyager
from loss_fns.hierarchical_ce import HierarchicalCrossEntropyWithLogitsLoss


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

    dataloader = benchmark.split()

    num_offsets = 1 << config.offset_bits
    criterion = HierarchicalCrossEntropyWithLogitsLoss(
        multi_label=config.multi_label, num_offsets=num_offsets
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    # scheduler = ExponentialLR(optimizer, gamma=0.95)

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
        total_page_correct = 0
        total_offset_correct = 0
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

            if batch % 1000 == 0 and batch != 0:
                ms_per_batch = (time.time() - start_time) * 1000 / batch
                print(
                    f"epoch {epoch+1} | batch {batch}/{len(dataloader)} batches | ms/batch {ms_per_batch} |  \
                    loss {total_loss:.4f} | page_acc {total_page_correct / (batch * args.batch_size) * 100:.4f} | offset_acc {total_offset_correct / (batch * args.batch_size) * 100:.4f}"
                )
                # total_page_correct = 0
                # total_offset_correct = 0
                # total_loss = 0
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


def count_page_correct(y_page, outputs, num_offsets, config):
    y_page_labels = y_page[:, -1]
    if config.sequence_loss:
        outputs = outputs[:, -1]
    if config.multi_label:
        pass
    #     y_page = multi_one_hot(y_page_labels, tf.shape(outputs)[-1] - self.num_offsets)
    #     page_correct = (y_page > 0.5) & (outputs[:, :-self.num_offsets] >= 0)
    else:
        # Compare labels against argmax
        page_correct = (y_page_labels == outputs[:, :-num_offsets].argmax(dim=-1)).int()
    return page_correct.sum().item()


def count_offset_correct(y_offset, outputs, num_offsets, config):
    y_offset_labels = y_offset[:, -1]
    if config.sequence_loss:
        outputs = outputs[:, -1]
    if config.multi_label:
        pass
        # y_offset = multi_one_hot(y_offset_labels, self.num_offsets)
        # offset_correct = (y_offset > 0.5) & (y_pred[:, -self.num_offsets:] >= 0)
    else:
        # Compare labels against argmax
        offset_correct = y_offset_labels == torch.argmax(
            outputs[:, -num_offsets:], dim=-1
        )
    return offset_correct.sum().item()


if __name__ == "__main__":
    args = parse_args()
    model = train(args)
