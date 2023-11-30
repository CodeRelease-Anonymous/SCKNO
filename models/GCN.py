# !/usr/bin/env python
# -*- coding:utf-8 -*-

from __future__ import division
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from tools.utils import sym_norm_Adj


class gconv(nn.Module):

    def __init__(self):
        super(gconv, self).__init__()

    def forward(self, A, x, shape_len):
        if shape_len == 3:
            if len(A.shape) == 2:
                x = torch.einsum('hw, bwc->bhc', (A, x))
            else:
                x = torch.einsum('bhw, bwc->bhc', (A, x))
        elif shape_len == 4:
            if len(A.shape) == 2:
                x = torch.einsum('hw, bwtc->bhtc', (A, x))
            else:
                x = torch.einsum('bhw, bwtc->bhtc', (A, x))
        return x.contiguous()


class linear(nn.Module):

    def __init__(self, c_in, c_out, bias=True):
        super(linear, self).__init__()
        self.mlp = nn.Linear(c_in, c_out, bias)

    def forward(self, x):
        return F.leaky_relu(self.mlp(x), inplace=True)


class mixpropGCN(nn.Module):

    def __init__(self, in_dim, out_dim, gdep, dropout_prob=0, alpha=0.3, norm_adj=None, relation_num=2, prop_function=None):
        super(mixpropGCN, self).__init__()
        self.nconv = gconv()
        self.mlp = linear((gdep + 1) * in_dim, out_dim)
        self.gdep = gdep
        self.dropout_prob = dropout_prob
        self.alpha = alpha
        self.norm_adj = norm_adj
        self.relation_num = relation_num
        self.prop_function = prop_function
        self.alpha=alpha

    def forward(self, x, norm_adj=None):
        if norm_adj == None:
            norm_adj = self.norm_adj
        h = x
        out = [x]

        for i in range(self.gdep):
            if self.prop_function is None:
                h = self.alpha * x + (1 - self.alpha) * self.nconv(norm_adj, h, len(x.shape))
            else:
                h = self.alpha * x + (1 - self.alpha) * self.prop_function(h, norm_adj)
            out.append(h)
        ho = torch.cat(out, dim=-1)
        ho = self.mlp(ho)
        if self.dropout_prob > 0:
            ho = F.dropout(ho, self.dropout_prob)
        return ho

class GMulitProp(nn.Module):
    def __init__(self, relation_num):
        super(GMulitProp, self).__init__()
        self.relation_num = relation_num
        self.weight = nn.Parameter(torch.ones(self.relation_num) / self.relation_num, requires_grad=True)

    def forward(self, x, graph_list):
        node_agg_effect = 0
        weight = F.softmax(self.weight, dim=-1)
        for i in range(len(graph_list)):
            if len(x.shape) >3:
                if len(graph_list[i].shape) == 3:
                    node_agg_effect += weight[i] * torch.einsum("bnm, bmtd->bntd", (graph_list[i], x))
                else:
                    node_agg_effect += weight[i] * torch.einsum("nm, bmtd->bntd", (graph_list[i], x))
            else:
                if len(graph_list[i].shape) == 3:
                    node_agg_effect += weight[i] * torch.einsum("bnm, bmd->bnd", (graph_list[i], x))
                else:
                    node_agg_effect += weight[i] * torch.einsum("nm, bmd->bnd", (graph_list[i], x))


        return node_agg_effect

