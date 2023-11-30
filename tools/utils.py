#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import torch
import numpy as np
import pickle
import torch.nn as nn
import torch.nn.functional as F
import scipy.sparse as sparse
import matplotlib.pyplot as plt
import time
from datetime import datetime
import os
import math
import mkl_random
from scipy.sparse import linalg
from torch.optim.lr_scheduler import MultiStepLR
import colorsys
import random


class StepLR2(MultiStepLR):
    """StepLR with min_lr"""

    def __init__(self,
                 optimizer,
                 milestones,
                 gamma=0.1,
                 last_epoch=-1,
                 min_lr=2.0e-6):
        self.optimizer = optimizer
        self.milestones = milestones
        self.gamma = gamma
        self.last_epoch = last_epoch
        self.min_lr = min_lr
        super(StepLR2, self).__init__(optimizer, milestones, gamma)

    def get_lr(self):
        lr_candidate = super(StepLR2, self).get_lr()
        if isinstance(lr_candidate, list):
            for i in range(len(lr_candidate)):
                lr_candidate[i] = max(self.min_lr, lr_candidate[i])

        else:
            lr_candidate = max(self.min_lr, lr_candidate)

        return lr_candidate


class StandardScaler_Torch:
    """
    Standard the input
    """

    def __init__(self, mean, std, device):
        self.mean = torch.tensor(data=mean, dtype=torch.float, device=device)
        self.std = torch.tensor(data=std, dtype=torch.float, device=device)

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


def sym_norm_Adj(W):
    assert W.shape[0] == W.shape[1]

    N = W.shape[0]
    D = np.sum(W, axis=1)
    D = np.diag(D ** -0.5)
    D[np.isnan(D)] = 0.
    D[np.isinf(D)] = 0.
    sym_norm_Adj_matrix = np.dot(D, W)
    sym_norm_Adj_matrix = np.dot(sym_norm_Adj_matrix, D)

    return sym_norm_Adj_matrix


def asym_norm_Adj(W):
    assert W.shape[0] == W.shape[1]

    N = W.shape[0]
    D = np.diag(1.0 / np.sum(W, axis=1))  #
    D[np.isinf(D)] = 0.
    D[np.isnan(D)] = 0.
    norm_Adj_matrix = np.dot(D, W)  #

    return norm_Adj_matrix  #


def sym_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sparse.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isnan(d_inv_sqrt)] = 0.
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sparse.diags(d_inv_sqrt)
    sym_norm_Adj_matrix = np.dot(d_mat_inv_sqrt, adj)
    sym_norm_Adj_matrix = np.dot(sym_norm_Adj_matrix, d_mat_inv_sqrt)
    return sym_norm_Adj_matrix.astype(np.float32).todense()


def asym_adj(adj):
    """Asymmetrically normalize adjacency matrix."""
    adj = sparse.coo_matrix(adj)
    rowsum = np.array(adj.sum(1)).flatten()
    d_inv = np.power(rowsum, -1).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    d_inv[np.isnan(d_inv)] = 0.
    d_mat = sparse.diags(d_inv)
    return d_mat.dot(adj).astype(np.float32).todense()


def kl_normal_log(mu, logvar, mu_prior, logvar_prior):
    var = logvar.exp()
    var_prior = logvar_prior.exp()

    element_wise = 0.5 * (
                torch.log(var_prior) - torch.log(var) + var / var_prior + (mu - mu_prior).pow(2) / var_prior - 1)
    kl = element_wise.mean(-1)  #

    return kl.mean()


def make_saved_dir(saved_dir, use_time=3):
    """
    :param saved_dir:
    :return: {saved_dir}/{%m-%d-%H-%M-%S}
    """
    if not os.path.exists(saved_dir):
        os.makedirs(saved_dir)
    if use_time == 1:
        saved_dir = os.path.join(saved_dir, datetime.now().strftime('%m-%d_%H:%M'))
        if not os.path.exists(saved_dir):
            os.makedirs(saved_dir)
    elif use_time == 2:
        saved_dir = os.path.join(saved_dir, datetime.now().strftime('%m-%d'))
        if not os.path.exists(saved_dir):
            os.makedirs(saved_dir)

    return saved_dir


def chrono_embedding_sincos(chrono_arr, periods):
    assert chrono_arr.shape[-1] == len(periods)
    batch, time = chrono_arr.shape[:2]
    periods = torch.tensor(periods).to(torch.float32).to(chrono_arr.device)
    t = chrono_arr / periods * 2 * np.pi
    sint, cost = torch.sin(t), torch.cos(t)
    sincos_emb = torch.stack([sint, cost], dim=-1)
    sincos_emb = sincos_emb.reshape(batch, time, -1)
    return sincos_emb
