#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

# MLP encoder/decoder
from torch.distributions import MultivariateNormal, Normal

class NLayerLeakyMLP(nn.Module):

    def __init__(self, in_features, out_features, num_layers, hidden_dim=64, bias=True):
        super().__init__()
        layers = []
        for l in range(num_layers):
            if l == 0:
                layers.append(nn.Linear(in_features, hidden_dim))  # 8->128
                layers.append(nn.LeakyReLU(0.2, True))
            else:
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                layers.append(nn.LeakyReLU(0.2, True))

        layers.append(nn.Linear(hidden_dim, out_features))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class Mapping(nn.Module):
    def __init__(self,
                 encoder_dim: int,
                 dist_dim: int,
                 hidden_dims=None,
                 ) -> None:
        super(Mapping, self).__init__()
        self.fc_mu = nn.Linear(encoder_dim, dist_dim)
        self.fc_logvar = nn.Linear(encoder_dim, dist_dim)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.LSTMCell):
                m.weight_hh.data.normal_(0, 0.1)
                m.weight_ih.data.normal_(0, 0.1)
                m.bias_hh.data.zero_()
                m.bias_ih.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.1)
                m.bias.data.zero_()

    def forward(self, hidden_states, use_prior=True):
        mus = self.fc_mu(hidden_states)
        logvars = self.fc_logvar(hidden_states)
        if use_prior:
            ps = MultivariateNormal(mus, torch.diag_embed(torch.exp(logvars*0.5) + 1e-16))
        else:
            ps = Normal(mus, torch.exp(logvars * 0.5) + 1e-16)       #
        return mus, logvars, ps