class TerribleMLModel(MLPrefetchModel):
    
    degree = 2
    k = int(os.environ.get('CNN_K', '2'))
    model_class = eval(os.environ.get('CNN_MODEL_CLASS', 'MLP'))
    history = int(os.environ.get('CNN_HISTORY', '4'))
    lookahead = int(os.environ.get('LOOKAHEAD', '5'))
    bucket = os.environ.get('BUCKET', 'ip')
    epochs = int(os.environ.get('EPOCHS', '30'))
    lr = float(os.environ.get('CNN_LR', '0.002'))
    window = history + lookahead + k
    filter_window = lookahead * degree
    next_page_table = defaultdict(dict)
    batch_size = 256

    def __init__(self):
        self.model = self.model_class()

    def load(self, path):
        self.model.load_state_dict(torch.load(path))

    def save(self, path):
        torch.save(self.model.state_dict(), path)

    def batch(self, data, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size
        bucket_data = defaultdict(list)
        bucket_instruction_ids = defaultdict(list)
        batch_instr_id, batch_page, batch_next_page, batch_x, batch_y = [], [], [], [], []
        for line in data:
            instr_id, cycles, load_address, ip, hit = line
            page = load_address >> 12
            ippage = (ip, page)
            bucket_key = eval(self.bucket)
            bucket_buffer = bucket_data[bucket_key]
            bucket_buffer.append(load_address)
            bucket_instruction_ids[bucket_key].append(instr_id)
            if len(bucket_buffer) >= self.window:
                current_page = bucket_buffer[self.history - 1] >> 12
                last_page = bucket_buffer[self.history - 2] >> 12
                if last_page != current_page:
                    self.next_page_table[ip][last_page] = current_page
                batch_page.append(bucket_buffer[self.history - 1] >> 12)
                batch_next_page.append(self.next_page_table[ip].get(current_page, current_page))
                # TODO send transition information for labels to represent
                batch_x.append(self.represent(bucket_buffer[:self.history], current_page))
                batch_y.append(self.represent(bucket_buffer[-self.k:], current_page, box=False))
                batch_instr_id.append(bucket_instruction_ids[bucket_key][self.history - 1])
                bucket_buffer.pop(0)
                bucket_instruction_ids[bucket_key].pop(0)
            if len(batch_x) == batch_size:
                if torch.cuda.is_available():
                    yield batch_instr_id, batch_page, batch_next_page, torch.Tensor(batch_x).cuda(), torch.Tensor(batch_y).cuda()
                else:
                    yield batch_instr_id, batch_page, batch_next_page, torch.Tensor(batch_x), torch.Tensor(batch_y)
                batch_instr_id, batch_page, batch_next_page, batch_x, batch_y = [], [], [], [], []

    def accuracy(self, output, label):
        return torch.sum(
            torch.logical_and(
                torch.scatter(
                    torch.zeros(output.shape, device=output.device), 1, torch.topk(output, self.k).indices, 1
                ),
                label
            )
        ) / label.shape[0] / self.k

    def train(self, data):
        print('LOOKAHEAD =', self.lookahead)
        print('BUCKET =', self.bucket)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        # defining the loss function
        # criterion = nn.CrossEntropyLoss()
        criterion = nn.BCELoss()
        # checking if GPU is available
        if torch.cuda.is_available():
            self.model = self.model.cuda()
            criterion = criterion.cuda()
        # converting the data into GPU format
        self.model.train()

        for epoch in range(self.epochs):
            accs = []
            losses = []
            percent = len(data) // self.batch_size // 100
            for i, (instr_id, page, next_page, x_train, y_train) in enumerate(self.batch(data)):
                # clearing the Gradients of the model parameters
                optimizer.zero_grad()

                # prediction for training and validation set
                output_train = self.model(x_train)

                # computing the training and validation loss
                loss_train = criterion(output_train, y_train)
                acc = self.accuracy(output_train, y_train)
                # print('Acc {}: {}'.format(epoch, acc))

                # computing the updated weights of all the model parameters
                loss_train.backward()
                optimizer.step()
                tr_loss = loss_train.item()
                accs.append(float(acc))
                losses.append(float(tr_loss))
                if i % percent == 0:
                    print('.', end='')
            print('Acc {}: {}'.format(epoch, sum(accs) / len(accs)))
            print('Epoch : ', epoch + 1, '\t', 'loss :', sum(losses))

    def generate(self, data):
        self.model.eval()
        prefetches = []
        accs = []
        order = {i: line[0] for i, line in enumerate(data)}
        reverse_order = {v: k for k, v in order.items()}
        for i, (instr_ids, pages, next_pages, x, y) in enumerate(self.batch(data)):
            # breakpoint()
            pages = torch.LongTensor(pages).to(x.device)
            next_pages = torch.LongTensor(next_pages).to(x.device)
            instr_ids = torch.LongTensor(instr_ids).to(x.device)
            y_preds = self.model(x)
            accs.append(float(self.accuracy(y_preds, y)))
            topk = torch.topk(y_preds, self.degree).indices
            shape = (topk.shape[0] * self.degree,)
            topk = topk.reshape(shape)
            pages = torch.repeat_interleave(pages, self.degree)
            next_pages = torch.repeat_interleave(next_pages, self.degree)
            instr_ids = torch.repeat_interleave(instr_ids, self.degree)
            addresses = (topk < 64) * (pages << 12) + (topk >= 64) * ((next_pages << 12) - (64 << 6)) + (topk << 6)
            #addresses = (pages << 12) + (topk << 6)
            prefetches.extend(zip(map(int, instr_ids), map(int, addresses)))
            if i % 100 == 0:
                print('Chunk', i, 'Accuracy', sum(accs) / len(accs))
        prefetches = sorted([(reverse_order[iid], iid, addr) for iid, addr in prefetches])
        prefetches = [(iid, addr) for _, iid, addr in prefetches]
        return prefetches

    def represent(self, addresses, first_page, box=True):
        blocks = [(address >> 6) % 64 for address in addresses]
        pages = [(address >> 12) for address in addresses]
        raw = [0 for _ in range(128)]
        for i, block in enumerate(blocks):
            if first_page == pages[i]:
                raw[block] = 1
            else:
                raw[64 + block] = 1
        if box:
            return [raw]
        else:
            return raw
