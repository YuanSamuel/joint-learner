import torch
import torch.nn as nn
import torch.nn.functional as F

class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        # Calculate Euclidean distance between the output embeddings
        dist = F.pairwise_distance(output1, output2)

        # Contrastive loss calculation
        # loss_similar = (1 - label) * 0.5 * torch.pow(dist, 2)
        # loss_dissimilar = label * 0.5 * torch.pow(torch.clamp(self.margin - dist, min=0.0), 2)
        # print(dist, label, loss_similar, loss_dissimilar)

        # loss = torch.mean(loss_similar + loss_dissimilar)
        # print(output1.shape, output2.shape, label.shape, dist)
        # print(label)
        # print((label) * torch.pow(dist, 2))
        # print(torch.pow(torch.clamp(self.margin - dist, min=0.0), 2))

        loss = torch.mean((label) * torch.pow(dist, 2) +
                                (1 - label) * torch.pow(torch.clamp(self.margin - dist, min=0.0), 2))
        
        # print(loss)

        return loss
