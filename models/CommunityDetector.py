#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import torch
from torch import nn
import torch.distributions as D
import torch.nn.init as init
from torch.distributions import MultivariateNormal
from torch.nn import functional as F
class CommunityDetector(nn.Module):
    def __init__(self,
                 node: int = 50,
                 time: int = 12,
                 input_dim: int = 10,
                 attr_dim = 10,
                 control_dim=54,
                 hidden_dim = 64,
                 community_num: int = 20,
                 community_embedding_dim=32,
                 ):
        super().__init__()

        self.fusion = nn.Linear(input_dim+control_dim, hidden_dim)
        self.domain_embedding = nn.Parameter(torch.randn(community_num, community_embedding_dim), requires_grad=True)
        nn.init.orthogonal_(self.domain_embedding)

        self.detector = nn.Linear(time*hidden_dim, community_embedding_dim)

    def forward(self, states, controls=None):
        batch, node, time, dim = states.shape
        if controls is not None:
            latent = self.fusion(torch.cat([states, controls], dim=-1))
        else:
            latent = self.fusion(states)

        tmp = latent.mean(0)
        latent = self.detector(tmp.reshape(node, -1))
        community = torch.matmul(latent, self.domain_embedding.T)
        community = F.gumbel_softmax(community, tau=1, hard=True, dim=-1)

        return community


    def embedding_constrains(self):
        mat = torch.abs(torch.mm(self.domain_embedding, self.domain_embedding.T))
        diversity_loss = (mat - torch.eye(mat.shape[0]).to(mat.device))**2
        diversity_loss = torch.norm(diversity_loss, p=2)
        return diversity_loss