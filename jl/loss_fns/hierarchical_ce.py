import torch
import torch.nn as nn
import torch.nn.functional as F

class HierarchicalCrossEntropyWithLogitsLoss(nn.Module):
    '''
    Hierarchical CrossEntropy loss that Voyager was trained with in original paper
    '''
    def __init__(self, multi_label, num_offsets=64):
        super(HierarchicalCrossEntropyWithLogitsLoss, self).__init__()
        self.multi_label = multi_label
        self.num_offsets = num_offsets
        if not self.multi_label:
            self.cross_entropy = nn.CrossEntropyLoss()
        
    def forward(self, y_pred, y_true):
        if self.multi_label:
            # Extra time access with only one timestep
            y_page_labels = y_true[0].squeeze(1).squeeze(1)
            y_offset_labels = y_true[1].squeeze(1).squeeze(1)

            print("TESTING")
            print(y_true[0])
            print(y_page_labels)
            print(y_page_labels)
            print(y_true[0].shape, y_page_labels.shape)
            
            # Create one-hot representation for cross_entropy
            y_page = F.one_hot(y_page_labels, num_classes=y_pred.size(-1) - self.num_offsets).float()
            y_offset = F.one_hot(y_offset_labels, num_classes=self.num_offsets).float()

            page_loss = F.binary_cross_entropy_with_logits(y_pred[:, :-self.num_offsets], y_page, reduction='mean')
            offset_loss = F.binary_cross_entropy_with_logits(y_pred[:, -self.num_offsets:], y_offset, reduction='mean')
        else:
            # print(y_pred[:, :-self.num_offsets], y_true[0].squeeze(1))
            # print(y_pred[:, -self.num_offsets:], y_true[1].squeeze(1))
            page_loss = self.cross_entropy(y_pred[:, :-self.num_offsets], y_true[0].squeeze(1))
            offset_loss = self.cross_entropy(y_pred[:, -self.num_offsets:], y_true[1].squeeze(1))

        return page_loss + offset_loss
    