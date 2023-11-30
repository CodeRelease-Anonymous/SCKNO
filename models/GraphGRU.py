# !/usr/bin/env python
# -*- coding:utf-8 -*-
import torch
import torch.nn as nn

from models.GCN import mixpropGCN
from tools.utils import sym_adj


class GraphGRU(nn.Module):

    def __init__(self,
                 in_dim,
                 hidden_dim,
                 # norm_adj,
                 gcn_depth=2,
                 dropout_type='dropout',
                 dropout_prob=0.3,
                 alpha=0.3,
                 prop_function=None):
        super(GraphGRU, self).__init__()

        self.in_channels = in_dim
        self.hidden_dim = hidden_dim
        self.dropout_type = dropout_type
        self.dropout_prob = dropout_prob
        self.dropout = nn.Dropout(dropout_prob)

        self.GCN_update = mixpropGCN(in_dim+hidden_dim, hidden_dim, gcn_depth, dropout_prob, alpha=alpha,
                                     prop_function=prop_function)
        self.GCN_reset = mixpropGCN(in_dim+hidden_dim, hidden_dim, gcn_depth, dropout_prob, alpha=alpha,
                                    prop_function=prop_function)
        self.GCN_cell = mixpropGCN(in_dim+hidden_dim, hidden_dim, gcn_depth, dropout_prob, alpha=alpha,
                                   prop_function=prop_function)

        self.layerNorm = nn.LayerNorm([self.hidden_dim])


    def forward(self, inputs, hidden_state=None, norm_adj=None):
        '''
        GraphGRU
        :param inputs:
        :param hidden_state:
        :return:
        '''
        batch_size, node_num, in_dim = inputs.shape
        if hidden_state == None:
            hidden_state = torch.randn((batch_size, node_num, self.hidden_dim)).to(inputs.device)

        combined = torch.cat((inputs, hidden_state), dim=-1)

        update_gate = torch.sigmoid(self.GCN_update(combined, norm_adj))

        reset_gate = torch.sigmoid(self.GCN_reset(combined, norm_adj))

        temp = torch.cat((inputs, torch.mul(reset_gate, hidden_state)), dim=-1)
        cell_State = torch.tanh(self.GCN_cell(temp, norm_adj))         #

        next_Hidden_State = torch.mul(update_gate, hidden_state) + torch.mul(1.0 - update_gate, cell_State)
        next_hidden = self.layerNorm(next_Hidden_State)

        output = next_hidden
        return output, next_hidden
