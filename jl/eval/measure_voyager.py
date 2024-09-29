import torch


def count_page_correct(y_page, outputs, num_offsets, config):
    y_page_labels = y_page[:, -1]
    if config.sequence_loss:
        outputs = outputs[:, -1]
    if config.multi_label:
        # pass
        y_page = torch.zeros(y_page_labels.shape + (outputs.size(-1) - num_offsets,), dtype=torch.float32)
        y_page.scatter_(-1, y_page_labels.unsqueeze(-1), 1.0)
        page_correct = (y_page > 0.5) & (outputs[:, :-num_offsets] >= 0)
        # y_page = multi_one_hot(y_page_labels, tf.shape(outputs)[-1] - num_offsets)
        # page_correct = (y_page > 0.5) & (outputs[:, :-num_offsets] >= 0)
    else:
        # Compare labels against argmax
        page_correct = (y_page_labels == outputs[:, :-num_offsets].argmax(dim=-1)).int()
    return page_correct.sum().item()


def count_offset_correct(y_offset, outputs, num_offsets, config):
    y_offset_labels = y_offset[:, -1]
    if config.sequence_loss:
        outputs = outputs[:, -1]
    if config.multi_label:
        y_offset = torch.zeros(y_offset_labels.shape + (outputs.size(-1) - num_offsets,), dtype=torch.float32)
        y_offset.scatter_(-1, y_offset_labels.unsqueeze(-1), 1.0)
        offset_correct = (y_offset > 0.5) & (outputs[:, -num_offsets:] >= 0)
        # y_offset = multi_one_hot(y_offset_labels, self.num_offsets)
        # offset_correct = (y_offset > 0.5) & (y_pred[:, -self.num_offsets:] >= 0)
    else:
        # Compare labels against argmax
        offset_correct = (y_offset_labels == outputs[:, -num_offsets:].argmax(dim=-1)).int()
    return offset_correct.sum().item()


def count_overall_correct(y_page, y_offset, outputs, num_offsets, config):
    y_page_labels = y_page[:, -1]
    y_offset_labels = y_offset[:, -1]
    if config.sequence_loss:
        outputs = outputs[:, -1]
    if config.multi_label:
        pass
    else:
        # Compare labels against argmax
        page_correct = (y_page_labels == outputs[:, :-num_offsets].argmax(dim=-1)).int()
        offset_correct = (y_offset_labels == outputs[:, -num_offsets:].argmax(dim=-1)).int()
    return (page_correct & offset_correct).sum().item()
