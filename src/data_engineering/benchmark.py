from collections import defaultdict
import lzma
from random import randint
import time
import torch
from torch.utils.data import Dataset, DataLoader


# Used to decorate functions for timing purposes
def timefunction(text=""):
    # Need to do a double decorate since we want the text parameter
    def decorate(f):
        def g(*args, **kwargs):
            start = time.time()
            print(text + "...", end="")
            ret = f(*args, **kwargs)
            end = time.time()
            print("Done in", end - start, "seconds")
            return ret

        return g

    # Returns the decorating function with the text parameter available via closure
    return decorate


class PrefetcherDataset(Dataset):
    def __init__(self, data, start_idx, end_idx, transform=None):
        self.data = data[start_idx:end_idx]
        self.start_idx = start_idx
        self.end_idx = end_idx
        self.transform = transform

    def __len__(self):
        return self.end_idx - self.start_idx

    def __getitem__(self, idx):
        sample = self.data[idx]
        if self.transform:
            sample = self.transform(idx)
        return sample


class BenchmarkTrace:
    """
    Benchmark trace parsing class
    """

    def __init__(self, config, args):
        self.config = config
        self.pc_mapping = {"oov": 0}
        self.page_mapping = {"oov": 0}
        self.reverse_page_mapping = {}
        self.data = [[0, 0, 0, 0, 0]]
        self.pc_data = [[]]
        self.online_cutoffs = []
        # Boolean value indicating whether or not to use the multiple labeling scheme
        self.offset_mask = (1 << config.offset_bits) - 1
        # Stores pc localized streams
        self.pc_addrs = {}
        self.pc_addrs_idx = {}
        # Counts number of occurences for low frequency addresses for delta prediction
        self.count = {}
        # Stores address localized streams for spatial localization
        self.cache_lines = {}
        self.cache_lines_idx = {}
        self.orig_addr = [0]
        self.train_split = None
        self.valid_split = None
        self.batch_size = args.batch_size

    def __len__(self):
        return len(self.indices)

    def read_and_process_file(self, f):
        self._read_file(f)
        self.reverse_page_mapping = {v: k for k, v in self.page_mapping.items()}
        if self.config.use_deltas:
            self._replace_with_deltas()
            # Needs to be regenerated to include deltas
            self.reverse_page_mapping = {v: k for k, v in self.page_mapping.items()}
        if self.config.multi_label:
            self._generate_multi_label()

        self._tensor()

    @timefunction("Tensoring data")
    def _tensor(self):
        self.data = torch.as_tensor(self.data)
        if self.config.multi_label:
            self.pages = torch.as_tensor(self.pages)
            self.offsets = torch.as_tensor(self.offsets)

        self.pc_data = [torch.as_tensor(item) for item in self.pc_data]

    @timefunction("Reading in data")
    def _read_file(self, f):
        """
        Reads and processes the data in the benchmark trace files
        """
        cur_phase = 1
        phase_size = 50 * 1000 * 1000
        for i, line in enumerate(f):
            # Necessary for some extraneous lines in MLPrefetchingCompetition traces
            if line.startswith("***") or line.startswith("Read"):
                continue
            inst_id, pc, addr = self.process_line(line)
            if self.train_split is None and inst_id >= 200 * 1000 * 1000:
                self.train_split = i
            elif self.valid_split is None and inst_id >= 225 * 1000 * 1000:
                self.valid_split = i
            if inst_id >= 250 * 1000 * 1000:
                break
            # We want 1 epoch per 50M instructions
            # TODO: Do we want to do every 50M instructions or 50M load instructions?
            if inst_id >= cur_phase * phase_size:
                self.online_cutoffs.append(i)
                cur_phase += 1
            self.process_row(i, inst_id, pc, addr)

    def _replace_with_deltas(self):
        prev_addr = {}
        if self.config.global_output:
            prev_addr["global"] = None
        n_deltas = 0
        n_total = 0
        n_applicable = 0
        self.orig = defaultdict(lambda: (0, 0))
        for i, (inst_id, mapped_pc, mapped_page, offset, pc_data_idx) in enumerate(
            self.data
        ):
            if i == 0:
                continue
            n_total += 1
            # Cache line of next load
            cur_addr = (
                self.reverse_page_mapping[mapped_page] << self.config.offset_bits
            ) + offset

            if self.count[(mapped_page, offset)] <= 2:
                n_applicable += 1
                # Only do delta if we're not the first memory address
                if (self.config.pc_localized and mapped_pc in prev_addr) or (
                    self.config.global_stream and prev_addr["global"] is not None
                ):
                    if self.config.pc_localized:
                        dist = cur_addr - prev_addr[mapped_pc]
                    else:
                        dist = cur_addr - prev_addr["global"]
                    # Only do deltas for pages within 256 pages, which as of right now
                    # experimentally looks like it gives good coverage without unnecessarily
                    # blowing up the vocabulary size
                    if 0 <= (abs(dist) >> self.config.offset_bits) <= 256:
                        dist_page = None
                        dist_offset = abs(dist) & self.offset_mask
                        if dist > 0:
                            dist_page = "+" + str(abs(dist) >> self.config.offset_bits)
                        elif dist < 0:
                            dist_page = "-" + str(abs(dist) >> self.config.offset_bits)

                        # We don't care about when the next address was the previous one
                        if dist_page is not None:
                            if dist_page not in self.page_mapping:
                                self.page_mapping[dist_page] = len(self.page_mapping)

                            self.orig[i] = [
                                self.reverse_page_mapping[self.data[i][2]],
                                self.data[i][3],
                            ]
                            self.data[i][2] = self.page_mapping[dist_page]
                            self.data[i][3] = dist_offset
                            n_deltas += 1

            # Save for delta
            if self.config.pc_localized:
                prev_addr[mapped_pc] = cur_addr
            else:
                prev_addr["global"] = cur_addr
        print("# Deltas:", n_deltas, n_deltas / n_applicable, n_deltas / n_total)

    def _apply_delta(self, addr, page, offset):
        dist = (int(self.reverse_page_mapping[page][1:]) << 6) + offset
        if self.reverse_page_mapping[page][0] == "+":
            return addr + dist
        else:
            return addr - dist

    def _apply_delta_to_idx(self, idx, page, offset):
        prev_idx = self.pc_data[self.data[idx, 1], self.data[idx, 4] - 1]
        prev_addr = self._idx_to_addr(prev_idx)
        return self._apply_delta(prev_addr, page, offset)

    def _idx_to_addr(self, data_idx):
        page = self.data[data_idx][2]
        actual_page = self.reverse_page_mapping[page]
        offset = self.data[data_idx][3]
        if isinstance(actual_page, str):
            return (self.orig[data_idx][0] << self.config.offset_bits) + self.orig[
                data_idx
            ][1]
        else:
            return (actual_page << self.config.offset_bits) + offset

    @timefunction("Generating multi-label data")
    def _generate_multi_label(self):
        # Previous address for delta localization
        self.pages = []
        self.offsets = []
        # 1 Global / Delta + 1 PC + 1 Spatial + up to 10 Co-occurrence
        width = 13
        for i, (inst_id, mapped_pc, mapped_page, offset, pc_data_idx) in enumerate(
            self.data
        ):
            if i == 0:
                self.pages.append([0 for _ in range(width)])
                self.offsets.append([0 for _ in range(width)])
                continue
            # First spot associated with main output
            mapped_pages = [mapped_page]
            offsets = [offset]

            cur_addr = self._idx_to_addr(i)

            # If we're PC localized, we do global here
            if not self.config.global_output:
                if i < len(self.data) - 1:
                    mapped_pages.append(self.data[i + 1][2])
                    offsets.append(self.data[i + 1][3])
                else:
                    mapped_pages.append(-1)
                    offsets.append(-1)
            else:
                """
                # PC LOCALIZATION
                if self.pc_addrs_idx[mapped_pc] < len(self.pc_addrs[mapped_pc]) - 1:
                    self.pc_addrs_idx[mapped_pc] += 1
                    pc_page, pc_offset = self.pc_addrs[mapped_pc][self.pc_addrs_idx[mapped_pc]]
                    mapped_pages.append(pc_page)
                    offsets.append(pc_offset)
                # Want to associate third spot with pc localization
                else:
                    mapped_pages.append(-1)
                    offsets.append(-1)
                """
                mapped_pages.append(-1)
                offsets.append(-1)

            # SPATIAL LOCALIZATION
            # TODO: Possibly need to examine this default distance value to make sure that it
            #       isn't too small (or large) for accessing the max temporal distance
            for j, (_, _, spatial_page, spatial_offset, _) in enumerate(
                self.data[i : i + 10]
            ):
                spatial_addr = self._idx_to_addr(j + i)
                if 0 < abs(spatial_addr - cur_addr) < 257:
                    mapped_pages.append(spatial_page)
                    offsets.append(spatial_offset)
                    break
            # Want to associate the fourth spot with spatial localization
            else:
                mapped_pages.append(-1)
                offsets.append(-1)

            # CO-OCCURRENCE
            # Compute frequency of pages in the future
            # TODO: Make sure that we want to do page instead of cache line.
            #       I tried using the cache line, but across the handful of
            #       benchmarks that I tested it on, there were maybe 30ish cases
            #       where the exact same cache line showed up more than once.
            #       - One possibility is that we might want to find the most
            #         occurring cache line given a history (of 1 or more) of the
            #         last loads throughout the trace.
            freq = {}
            best = 0
            for _, _, co_page, co_offset, _ in self.data[i : i + 10]:
                tag = (co_page, co_offset)
                if tag not in freq:
                    freq[tag] = 0
                freq[tag] += 1
                best = max(best, freq[tag])

            # Only want most frequent if it appears more than once, otherwise
            # co-occurrence would be slightly meaningless
            if best >= 2:
                # Take the most frequent cache lines
                for _, _, co_page, co_offset, _ in self.data[i : i + 10]:
                    if freq[(co_page, co_offset)] == best:
                        mapped_pages.append(co_page)
                        offsets.append(co_offset)

            # Working with rectangular tensors is infinitely more painless than
            # Tensorflow's ragged tensors that have a bunch of unsupported ops
            for i in range(len(mapped_pages), width):
                mapped_pages.append(-1)
                offsets.append(-1)

            # Append the final list of pages and offsets. There will be some
            # duplicates, but that's okay.
            self.pages.append(mapped_pages)
            self.offsets.append(offsets)

        # No longer need these, might as well free it up from memory
        del self.pc_addrs
        del self.pc_addrs_idx
        del self.cache_lines
        del self.cache_lines_idx
        del self.count

    def process_row(self, idx, inst_id, pc, addr):
        """
        Process PC / Address

        TODO: Handle the pc localization
        """
        cache_line = addr >> 6
        page, offset = (
            cache_line >> self.config.offset_bits,
            cache_line & self.offset_mask,
        )

        if pc not in self.pc_mapping:
            self.pc_mapping[pc] = len(self.pc_mapping)
            # These are needed for PC localization
            self.pc_addrs[self.pc_mapping[pc]] = []
            self.pc_addrs_idx[self.pc_mapping[pc]] = 0
            self.pc_data.append(
                [
                    0
                    for _ in range(
                        self.config.sequence_length + self.config.prediction_depth
                    )
                ]
            )

        if page not in self.page_mapping:
            self.page_mapping[page] = len(self.page_mapping)

        # Needed for delta localization
        if (self.page_mapping[page], offset) not in self.count:
            self.count[(self.page_mapping[page], offset)] = 0
        self.count[(self.page_mapping[page], offset)] += 1

        self.pc_addrs[self.pc_mapping[pc]].append((self.page_mapping[page], offset))

        # Needed for spatial localization
        if cache_line not in self.cache_lines:
            self.cache_lines[cache_line] = []
            self.cache_lines_idx[cache_line] = 0
        self.cache_lines[cache_line].append(idx)

        # Include the instruction ID for generating the prefetch file for running
        # in the ML-DPC modified version of ChampSim.
        # See github.com/Quangmire/ChampSim
        self.data.append(
            [
                inst_id,
                self.pc_mapping[pc],
                self.page_mapping[page],
                offset,
                len(self.pc_data[self.pc_mapping[pc]]),
            ]
        )
        self.orig_addr.append(cache_line)
        self.pc_data[self.pc_mapping[pc]].append(len(self.data) - 1)

    def process_line(self, line):
        # File format for ML Prefetching Competition
        # See github.com/Quangmire/ChampSim
        # Uniq Instr ID, Cycle Count,   Load Address,      PC of Load,        LLC Hit or Miss
        # int(split[0]), int(split[1]), int(split[2], 16), int(split[3], 16), split[4] == '1'

        # Return Inst ID, PC, and Load Address
        split = line.strip().split(", ")
        return (int(split[0]), int(split[3], 16), int(split[2], 16))

    def num_pcs(self):
        return len(self.pc_mapping)

    def num_pages(self):
        return len(self.page_mapping)

    def split(self, start_epoch=0, start_step=0, online=False, start_phase=0):
        """
        Splits the trace data into train / valid / test datasets
        """

        def mapper(idx):
            """
            Maps index in dataset to x = [pc_hist, page_hist, offset_hist], y = [page_target, offset_target]
            If sequence, page_target / offset_target is also a sequence

            Given a batch (x, y) where the first dimension corresponds to the batch
            pc_hist = x[:, :seq_len], page_hist = x[:, seq_len:2 * seq_len], offset_hist = x[:, 2 * seq_len:]
            page_target = y[:, 0], offset_target = y[:, 1]

            If sequence:
            page_target = y[:, 0, :], offset_target = y[:, 1, :]
            where the third axis / dimension is the time dimension
            """
            hists = []

            # PC Localized Input and Output
            cur_pc = self.data[idx, 1].item()
            end = self.data[idx, 4].item()
            start = end - self.config.sequence_length - self.config.prediction_depth

            if self.config.pc_localized:
                indices = self.pc_data[cur_pc][start : end + 1].long()
                hist = self.data[indices]
                page_hist = hist[: self.config.sequence_length, 2]
                offset_hist = hist[: self.config.sequence_length, 3]
                if self.config.use_current_pc:
                    pc_hist = hist[1 : self.config.sequence_length + 1, 1]
                else:
                    pc_hist = hist[: self.config.sequence_length, 1]
                hists.extend([pc_hist, page_hist, offset_hist])

            if not self.config.global_output:
                inst_id = self.data[self.pc_data[cur_pc][end - 1].item(), 0]
                if self.config.multi_label:
                    indices = self.pc_data[cur_pc][start : end + 1].long()
                    page_hist = self.pages[indices]
                    offset_hist = self.offsets[indices]
                    if self.config.sequence_loss:
                        y_page = page_hist[1 + self.config.prediction_depth :]
                        y_offset = offset_hist[1 + self.config.prediction_depth :]
                    else:
                        y_page = page_hist[-1:]
                        y_offset = offset_hist[-1:]
                else:
                    if self.config.sequence_loss:
                        y_page = hist[1 + self.config.prediction_depth :, 2]
                        y_offset = hist[1 + self.config.prediction_depth :, 3]
                    else:
                        y_page = hist[-1:, 2]
                        y_offset = hist[-1:, 3]

            return idx, inst_id, torch.cat(hists, dim=-1), y_page, y_offset

        # Closure for generating a reproducible random sequence
        # epoch_size = self.config.steps_per_epoch * self.config.batch_size

        # def random_closure(minval, maxval):
        #     def random(x):
        #         epoch = x // epoch_size
        #         step = x % epoch_size
        #         # Set the seed based on epoch and step for reproducibility
        #         torch.manual_seed(torch.tensor([epoch, step]).numpy().tobytes())
        #         # Generate a random integer
        #         return torch.randint(low=minval, high=maxval, size=())

        #     return random

        # if online:
        #     # Sets of train and eval datasets for each epoch
        #     train_datasets = []
        #     eval_datasets = []
        #     # Exclude the first N values since they cannot fully be an input for Voyager
        #     cutoffs = [
        #         self.config.sequence_length + self.config.prediction_depth
        #     ] + self.online_cutoffs

        #     # Include the rest of the data if it wasn't on a 50M boundary
        #     if self.online_cutoffs[-1] < len(self.data):
        #         self.online_cutoffs.append(len(self.data))

        #     # Stop before the last dataset since we don't need to train on it
        #     for i in range(start_phase, len(cutoffs) - 2):
        #         # Train on DATA[idx[i]] to DATA[idx[i + 1]]
        #         # Evaluate on DATA[idx[i + 1]] to DATA[idx[i + 2]]
        #         # Calculate the start index for the train dataset
        #         start_index = (
        #             start_epoch * epoch_size + start_step * self.config["batch_size"]
        #             if i == start_phase
        #             else 0
        #         )
        #         end_index = self.config["num_epochs_online"] * epoch_size

        #         if i == start_phase:
        #             start_index = start_epoch * epoch_size + start_step * self.config['batch_size']
        #             end_index = self.config['num_epochs_online'] * epoch_size
        #             train_dataset = PrefetcherDataset(self.data, start_index, cutoffs[i + 1],
        #                                         transform=random_closure(cutoffs[i], cutoffs[i + 1], epoch_size).map(self.mapper))
        #         else:
        #             train_dataset = PrefetcherDataset(self.data, cutoffs[i], cutoffs[i + 1],
        #                                         transform=random_closure(cutoffs[i], cutoffs[i + 1], epoch_size).map(self.mapper))

        #         eval_dataset = PrefetcherDataset(self.data, cutoffs[i + 1], cutoffs[i + 2], transform=self.mapper)

        #         train_datasets.append(DataLoader(train_dataset, batch_size=self.config['batch_size'], shuffle=True))
        #         eval_datasets.append(DataLoader(eval_dataset, batch_size=self.config['batch_size'], shuffle=False))


        #     return train_datasets, eval_datasets

        # Regular 80-10-10 decomposition
        train_split, valid_split = self.train_split, self.valid_split

        # Define the random closure with the specific range for training data
        # train_random_closure = random_closure(self.config.sequence_length + self.config.prediction_depth, train_split)
        train_mapper = mapper
        
        # Training Dataset
        # train_start = start_epoch * epoch_size + start_step * self.batch_size
        # train_end = self.config.num_epochs * epoch_size
        train_ds = PrefetcherDataset(self.data, 0, len(self.data), transform=lambda x: mapper(x))
        train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True, num_workers=4)

        # Validation Dataset
        # valid_ds = PrefetcherDataset(self.data, train_split, valid_split, transform=train_mapper)
        # valid_loader = DataLoader(valid_ds, batch_size=self.batch_size, num_workers=4)

        # # Test Dataset
        # test_ds = PrefetcherDataset(self.data, valid_split, len(self.data), transform=train_mapper)
        # test_loader = DataLoader(test_ds, batch_size=self.batch_size, num_workers=4)

        # return train_loader, valid_loader, test_loader

        return train_loader

    # Unmaps the page and offset
    def unmap(self, idx, x, page, offset, sequence_length):
        unmapped_page = self.reverse_page_mapping[page]

        # DELTA LOCALIZED
        if isinstance(unmapped_page, str):
            ret_addr = self._apply_delta_to_idx(idx, page, offset)
        else:
            ret_addr = (unmapped_page << self.config.offset_bits) + offset

        return ret_addr << 6

