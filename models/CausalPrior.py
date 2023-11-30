#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import torch
from torch import nn
import torch.distributions as D
import torch.nn.init as init
from torch.distributions import MultivariateNormal
from torch.nn import functional as F

from models.CausalTransitionFunction import CondAfflineCoupling
from models.MLP import NLayerLeakyMLP


class CausalPrior(nn.Module):
    def __init__(self,
                 node: int = 50,
                 input_dim: int = 10,
                 time: int = 6,
                 latent_dim: int = 8,
                 domain_num: int = 20,
                 domain_dim: int = 32,
                 hidden_dim: int = 64,
                 layer_num=2,
                 noise_dist_type='mlp',
                 base_dist_type='gaussian',
                 device='cpu',
                 tao = 1,
                 pror_function=None,
                 ):
        super().__init__()

        self.node = node
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim

        self.base_dist_type = base_dist_type
        self.noise_dist_type = noise_dist_type
        self.pror_function=pror_function
        self.device = device
        self.tao = tao
        self.causal_transition_function = CondAfflineCoupling(n_blocks=layer_num+1, input_size=latent_dim, hidden_size=hidden_dim,
                                                             layer_num=layer_num, condition_dim=latent_dim, batch_norm=False)


    def eig_decomp(self, K):
        """ eigen-decomp of K """
        # K = D*D
        eigenvalues, eigenvectors = torch.linalg.eig(K)
        eigenvectors_inv = torch.linalg.inv(eigenvectors)
        return eigenvalues, eigenvectors, eigenvectors_inv


    def causalGraph(self, koopman_tilde):

        eigenvalues, eigenvectors, eigenvectors_inv = self.eig_decomp(koopman_tilde)

        eigenvalues = eigenvalues + 1e-10 + 1e-10j
        remain_index1 = torch.where(eigenvalues.abs() <= 1.1)[0]
        remain_index2 = torch.where(eigenvalues.abs() >= 0.9)[0]
        stable = remain_index1[torch.isin(remain_index1, remain_index2)]
        if len(stable)<=0:
            stable = torch.arange(eigenvalues.shape[0])
        causal_eigenvalues = eigenvalues[stable]
        causal_eigenvectors = eigenvectors[:, stable]
        causal_eigenvectors_inv = eigenvectors_inv[stable, :]
        causal_graph = causal_eigenvectors@torch.diag_embed(causal_eigenvalues)@causal_eigenvectors_inv

        return causal_graph.real


    def noise_dist_prob(self, ez, x_domain_index=None, type='base', noise_dist=None):
        if type == 'base':
            prob_ez = noise_dist.log_prob(ez)
            return prob_ez


    def forward(self, latent, koopman_tilde, prior_graph=None, scale_type='global', community=None, noise_dist=None):

        batch, node, time, dim = latent.shape

        if scale_type == 'global':
            causalGraph = self.causalGraph(koopman_tilde)
            aggParent = latent@causalGraph
            ezs_est, det_jacobin = self.causal_transition_function(latent, aggParent)
            log_pz = self.noise_dist_prob(ezs_est, type='base', noise_dist=noise_dist) + det_jacobin

        elif scale_type == 'community':
            community_num = koopman_tilde.shape[0]
            community_class = torch.topk(community, k=1, dim=-1)[1].squeeze()
            parent = torch.zeros(batch, node, time, dim).to(latent.device)

            for i in range(community_num):
                causalGraph = self.causalGraph(koopman_tilde[i])
                community_latent = latent[:, community_class == i]                # B*C*TD
                aggParent = community_latent@causalGraph
                parent[:, community_class == i] = aggParent

            ezs_est, det_jacobin = self.causal_transition_function(latent, parent)
            log_pz = self.noise_dist_prob(ezs_est, x_domain_index=community, type='base',
                                          noise_dist=noise_dist) + det_jacobin

        elif scale_type == 'node':
            mean_koopman_tilde = koopman_tilde.mean(dim=0)
            causalGraph_list = []
            for n in range(len(mean_koopman_tilde)):
                causalGraph_n = self.causalGraph(mean_koopman_tilde[n])
                causalGraph_list.append(causalGraph_n)
            causalGraph = torch.stack(causalGraph_list, dim=0)
            aggParent = latent@causalGraph
            ezs_est, det_jacobin = self.causal_transition_function(latent, aggParent)
            log_pz = self.noise_dist_prob(ezs_est, type='base', noise_dist=noise_dist) + det_jacobin

        return log_pz


        
