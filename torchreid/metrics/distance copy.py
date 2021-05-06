from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import numpy as np

import torch
from torch.nn import functional as F

# distmat = metrics.compute_distance_matrix(qf, gf, qf_weight, gf_weight, dist_metric)
def compute_distance_matrix(input1, input2, input1_weight, input2_weight, metric='euclidean'):
    """A wrapper function for computing distance matrix.

    Args:
        input1 (torch.Tensor): 2-D feature matrix.
        input2 (torch.Tensor): 2-D feature matrix.
        input1_weight (torch.Tensor): 2-D feature weight matrix.
        input2_weight (torch.Tensor): 2-D feature weight matrix.
        metric (str, optional): "euclidean" or "cosine".
            Default is "euclidean".

    Returns:
        torch.Tensor: distance matrix.

    Examples::
       >>> from torchreid import metrics
       >>> input1 = torch.rand(10, 2048)
       >>> input2 = torch.rand(100, 2048)
       >>> distmat = metrics.compute_distance_matrix(input1, input2, input1_weight, input2_weight)
       >>> distmat.size() # (10, 100)
    """
    # check input
    assert isinstance(input1, torch.Tensor)
    assert isinstance(input2, torch.Tensor)
    assert input1.dim() == 2, 'Expected 2-D tensor, but got {}-D'.format(input1.dim())
    assert input2.dim() == 2, 'Expected 2-D tensor, but got {}-D'.format(input2.dim())
    assert input1.size(1) == input2.size(1)

    if metric == 'euclidean':
        distmat = euclidean_squared_distance(input1, input2, input1_weight, input2_weight)
    elif metric == 'cosine':
        distmat = cosine_distance(input1, input2, input1_weight, input2_weight)
    else:
        raise ValueError(
            'Unknown distance metric: {}. '
            'Please choose either "euclidean" or "cosine"'.format(metric)
        )
    
    return distmat


def euclidean_squared_distance(input1, input2, input1_weight, input2_weight):
    """Computes euclidean squared distance.

    Args:
        input1 (torch.Tensor): 2-D feature matrix. [m,d]
        input2 (torch.Tensor): 2-D feature matrix. [n,d]

    Returns:
        torch.Tensor: distance matrix. [m, n]
    """
    m, n = input1.size(0), input2.size(0)
    dist = torch.pow(input1, 2).sum(dim=1, keepdim=True).expand(m, n) + \
              torch.pow(input2, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    dist_g = torch.pow(input1, 2).sum(dim=1, keepdim=True).expand(m, n) + \
              torch.pow(input2, 2).sum(dim=1, keepdim=True).expand(n, m).t()

    input1 = input1.reshape([m, 5, -1])
    input2 = input2.reshape([n, 5, -1])
    input1_weight = input1_weight.reshape([m, 1, 4])
    input2_weight = input2_weight.reshape([1, n, 4])
    f_weight = input1_weight * input2_weight
    f_weight = torch.softmax(f_weight, dim=2)

    distmat_new = []
    for i in range(4):
        input1_i = input1[:, i+1, :]
        input1_i = input1_i.view(input1_i.size(0), -1)
        input2_i = input2[:, i+1, :]
        input2_i = input2_i.view(input2_i.size(0), -1)
        dist.addmm_(1, -2, input1_i, input2_i.t())
        distmat_new.append(dist)
    distmat = torch.stack(distmat_new, dim=2)
    distmat = distmat.cpu() * f_weight.cpu()
    distmat = torch.sum(distmat, dim=2)

    input1_g = input1[:, 0, :]
    input1_g = input1_g.view(input1_g.size(0), -1)
    input2_g = input2[:, 0, :]
    input2_g = input2_g.view(input2_g.size(0), -1)
    
    dist_g.addmm_(1, -2, input1_g, input2_g.t())
    distmat_last = dist_g + distmat 
    # print(distmat.size())
    return distmat_last


def cosine_distance(input1, input2, input1_weight, input2_weight):
    """Computes cosine distance.

    Args:
        input1 (torch.Tensor): 2-D feature matrix.
        input2 (torch.Tensor): 2-D feature matrix.

    Returns:
        torch.Tensor: distance matrix.
    """
    m, n = input1.size(0), input2.size(0)
    input1_normed = F.normalize(input1, p=2, dim=1)
    input2_normed = F.normalize(input2, p=2, dim=1)
    input1 = input1_normed.reshape([m, 6, -1])
    input2 = input2_normed.reshape([n, 6, -1])

    input1_weight = input1_weight.reshape([m, 1, 6])
    input2_weight = input2_weight.reshape([1, n, 6])
    f_weight = input1_weight * input2_weight
    f_weight = torch.softmax(f_weight, dim=2)
    # distmat = 1 - torch.mm(input1_normed, input2_normed.t())
    distmat_new = []
    for i in range(6):
        input1_i = input1[:, i, :]
        input1_i = input1_i.view(input1_i.size(0), -1)
        input2_i = input2[:, i, :]
        input2_i = input2_i.view(input2_i.size(0), -1)
        dist = 1 - torch.mm(input1_i, input2_i.t())
        distmat_new.append(dist)
    distmat = torch.stack(distmat_new, dim=2)
    distmat = distmat.cpu() * f_weight.cpu()
    distmat = torch.sum(distmat, dim=2)
    return distmat