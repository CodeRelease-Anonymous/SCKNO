#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from typing import (Tuple)
import torch.distributions as D


class FlowSequential(nn.Sequential):

    def forward(self, x, y):
        sum_log_abs_det_jacobians = 0
        for module in self:
            x, log_abs_det_jacobian = module(x, y)
            sum_log_abs_det_jacobians = sum_log_abs_det_jacobians + log_abs_det_jacobian
        return x, sum_log_abs_det_jacobians


class LinearMaskedCoupling(nn.Module):
    """
    Affline Coupling Transformation
    Modified RealNVP Coupling Layers per the MAF paper
    """

    def __init__(self, input_dim, hidden_dim, layer_num, mask, condition_dim=None):
        super().__init__()

        self.register_buffer('mask', mask)
        if condition_dim is not None:
            self.gate_FC_s = nn.Linear(condition_dim, input_dim)
            self.bias_FC_s = nn.Linear(condition_dim, input_dim)
            self.gate_FC_t = nn.Linear(condition_dim, input_dim)
            self.bias_FC_t = nn.Linear(condition_dim, input_dim)
        s_net = [nn.Linear(input_dim, hidden_dim)]
        for _ in range(layer_num):
            s_net += [nn.LeakyReLU(inplace=True), nn.Linear(hidden_dim, hidden_dim)]
        s_net += [nn.LeakyReLU(inplace=True), nn.Linear(hidden_dim, input_dim)]
        self.s_net = nn.Sequential(*s_net)
        # translation function
        self.t_net = copy.deepcopy(self.s_net)
        for i in range(len(self.t_net)):
            if not isinstance(self.t_net[i], nn.Linear):
                self.t_net[i] = nn.LeakyReLU(inplace=True)

    def forward(self, u, condition=None):
        mu = u * self.mask

        # run through model
        if condition is not None:
            batch, node, time, dim = mu.shape
            if len(condition.shape) == 3:
                gate_s = torch.sigmoid(self.gate_FC_s(condition)).unsqueeze(2)
                gate_t = torch.sigmoid(self.gate_FC_t(condition)).unsqueeze(2)
                gate_s_bias = self.bias_FC_s(condition).unsqueeze(2)
                gate_t_bias = self.bias_FC_t(condition).unsqueeze(2)

            else:
                gate_s = torch.sigmoid(self.gate_FC_s(condition))
                gate_t = torch.sigmoid(self.gate_FC_t(condition))
                gate_s_bias = self.bias_FC_s(condition)
                gate_t_bias = self.bias_FC_t(condition)

        s = self.s_net(mu)
        s = torch.mul(s, gate_s) + gate_s_bias
        s = torch.tanh(s) 

        t = self.t_net(mu)
        t = torch.mul(t, gate_t) + gate_t_bias

        x = mu + (1 - self.mask) * (u * s.exp() + t)

        log_abs_det_jacobian = (1 - self.mask) * s  # log det dx/du
        log_abs_det_jacobian = torch.sum(log_abs_det_jacobian, dim=-1)

        return x, log_abs_det_jacobian

class CondAfflineCoupling(nn.Module):

    def __init__(self, n_blocks, input_size, hidden_size, layer_num, condition_dim=None, batch_norm=True):
        super().__init__()
        # construct model
        modules = []
        mask = torch.arange(input_size).float() % 2
        for i in range(n_blocks):
            modules += [LinearMaskedCoupling(input_size, hidden_size, layer_num, mask, condition_dim)]
            mask = 1 - mask
        self.net = FlowSequential(*modules)

    def forward(self, x, condition=None):
        return self.net(x, condition)
