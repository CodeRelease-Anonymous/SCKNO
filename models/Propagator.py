#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F


class NodeEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(NodeEncoder, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, output_size),
        )

    def forward(self, x):

        return self.model(x)

class Propagator(nn.Module):
    def __init__(self, input_size, output_size, residual=False, norm_type='Layer'):
        super(Propagator, self).__init__()

        self.residual = residual

        self.linear = nn.Linear(input_size, output_size)
        if norm_type == 'Layer':
            self.norm = nn.LayerNorm(output_size)
        elif norm_type == 'Instance':
            self.norm = nn.InstanceNorm1d(output_size)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, res=None):
        if self.residual:
            x = self.relu(self.norm(self.linear(x) + res))
        else:
            x = self.relu(self.norm(self.linear(x)))

        return x

class ParticlePredictor(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(ParticlePredictor, self).__init__()

        self.linear_0 = nn.Linear(input_size, hidden_size)
        self.linear_1 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):

        x = self.relu(self.linear_0(x))

        return self.linear_1(x)


class PropagationNetwork(nn.Module):
    def __init__(self, attr_dim, state_dim, control_dim, prior_graph_dim,
                 node_hidden_dim, edge_hidden_dim, effect_hidden_dim, prop_function,
                 input_node_dim=None, input_edge_dim=None, output_dim=None, use_control=True, tanh=False,
                 residual=False, norm_type='Layer', gdep=2):

        super(PropagationNetwork, self).__init__()

        self.use_control = use_control
        self.gdep = gdep
        if input_node_dim is None:
            input_node_dim = attr_dim + state_dim
            input_node_dim += control_dim if use_control else 0

        if input_edge_dim is None:
            input_edge_dim = prior_graph_dim + state_dim

        if output_dim is None:
            output_dim = state_dim

        self.node_hidden_dim = node_hidden_dim
        self.edge_hidden_dim = edge_hidden_dim
        self.effect_hidden_dim = effect_hidden_dim

        self.residual = residual

        self.prop_function = prop_function
        self.node_encoder = NodeEncoder(input_node_dim, node_hidden_dim, effect_hidden_dim)
        self.node_propagator = [Propagator(2 * effect_hidden_dim, effect_hidden_dim, self.residual, norm_type)
                                for _ in range(gdep)]
        self.node_propagator = nn.ModuleList(self.node_propagator)
        self.node_predictor = ParticlePredictor(effect_hidden_dim, effect_hidden_dim, output_dim)

        if tanh:
            self.node_predictor = nn.Sequential(
                self.node_predictor, nn.Tanh()
            )

    def forward(self, attrs, states, controls=None, rel_attrs=None):
        B, N, T, in_dim = attrs.shape
        node_input_list = [attrs, states]
        if self.use_control:
            node_input_list += [controls]

        tmp = torch.cat(node_input_list, -1)                               # (B, node, time, attr+state)
        obj_encode = self.node_encoder(tmp)                                # (B, node, time, hidden)

        for i in range(self.gdep):
            if self.prop_function is None:
                if len(rel_attrs.shape)==2:
                    node_agg_effect = torch.einsum("nm, bmtd->bntd", (rel_attrs, obj_encode))
                else:
                    node_agg_effect = torch.einsum("bnm, bmtd->bntd", (rel_attrs, obj_encode))
            else:
                node_agg_effect = self.prop_function(obj_encode, rel_attrs)
            tmp = torch.cat([obj_encode, node_agg_effect], -1)
            obj_encode = self.node_propagator[i](tmp.reshape(B * N * T, -1)).reshape(B, N, T, -1)

        obj_prediction = self.node_predictor(obj_encode)

        return obj_prediction, 1
