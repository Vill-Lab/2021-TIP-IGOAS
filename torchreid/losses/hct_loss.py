from __future__ import absolute_import
from __future__ import division

import warnings

import torch
import torch.nn as nn


class HctLoss(nn.Module):
    """Center loss.
    
    Reference:
    Wen et al. A Discriminative Feature Learning Approach for Deep Face Recognition. ECCV 2016.
    
    Args:
    - num_classes (int): number of classes.
    - feat_dim (int): feature dimension.
    """
    def __init__(self, num_classes, feat_dim, margin, use_gpu=True):
        super(HctLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)
        self.use_gpu = use_gpu

        if self.use_gpu:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim).cuda())
        else:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim))

    def forward(self, inputs, targets):
        """
        Args:
        - inputs: feature matrix with shape (batch_size, feat_dim).
        - targets: ground truth labels with shape (num_classes).
        """
        batch_size = inputs.size(0)
        distmat = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + \
                  torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()
        distmat.addmm_(1, -2, inputs, self.centers.t())
        distmat = distmat.clamp(min=1e-12, max=1e+12)
        classes = torch.arange(self.num_classes).long()
        if self.use_gpu: classes = classes.cuda()
        targets = targets.unsqueeze(1).expand(batch_size, self.num_classes)
        mask = targets.eq(classes.expand(batch_size, self.num_classes))
        distnew = distmat.t()
        masknew = mask.t()
        dist_ap, dist_an = [], []
        for i in range(self.num_classes):
            if masknew[i].sum()==4:
                dist_ap.append(distnew[i][masknew[i]].max().unsqueeze(0))
                dist_an.append(distnew[i][masknew[i] == 0].min().unsqueeze(0))
            else: continue
        dist_ap = torch.cat(dist_ap)
        dist_an = torch.cat(dist_an)
        y = torch.ones_like(dist_an)
        loss = self.ranking_loss(dist_an, dist_ap, y)

        return loss
