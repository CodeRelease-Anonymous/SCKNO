#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import torch
from torch import nn
import torch.distributions as D
import torch.nn.functional as F

from models.GraphGRU import GraphGRU
from models.MLP import NLayerLeakyMLP


class GraphGRU_Encoder(nn.Module):

    def __init__(self,
                 node: int = 50,
                 num_for_predict: int = 12,
                 input_dim: int = 10,
                 attrs_dim: int = 4,
                 latent_dim: int = 8,
                 hidden_dim: int = 64,
                 gcn_depth:int=1,
                 alpha=0.3,
                 dropout_prob=0.4,
                 encoder_type='GraphGRU',
                 use_control=True,
                 random_sampling=True,
                 prop_function=None,
                 ):
        super().__init__()

        self.node = node
        self.time = num_for_predict
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.use_control = use_control
        self.posterior_type = encoder_type
        self.random_sampling = random_sampling

        self.input_encoder = NLayerLeakyMLP(in_features=input_dim,
                                            out_features=hidden_dim,
                                            num_layers=gcn_depth,
                                            hidden_dim=hidden_dim)

        dim = hidden_dim
        if encoder_type == 'GraphGRU':
            self.GraphGRU = GraphGRU(dim,
                                     hidden_dim,
                                     gcn_depth=gcn_depth,
                                     dropout_type='None',
                                     dropout_prob=dropout_prob,
                                     alpha=alpha,
                                     prop_function=prop_function)
    def log_qz(self, zs, mus, logvars):
        q_dist = D.Normal(mus, torch.exp(logvars * 0.5))  #
        log_qz = q_dist.log_prob(zs).sum(-1)

        return log_qz


    def forward(self, attrs, states, controls=None, rel_attrs=None):
        batch, node, time, in_dim = states.shape
        node_input_list = [attrs, states]
        if self.use_control:
            node_input_list += [controls]

        tmp = torch.cat(node_input_list, -1)  # (B, node, time, attr+state)
        x_feature = self.input_encoder(tmp)
        dists= []
        hidden_states = []
        hidden_state = torch.randn((batch, node, self.hidden_dim), device=states.device)
        for t in range(time):
            current_obs = x_feature[:, :, t, :]         # B, N, T, D

            output, hidden_state = self.GraphGRU(current_obs, hidden_state, rel_attrs)
            dists.append(output)
            hidden_states.append(hidden_state)
        dists = torch.stack(dists, dim=2)

        return dists, hidden_states
