#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast
import torch.distributions as D
from models.CausalPrior import CausalPrior
from models.CommunityDetector import CommunityDetector
from models.Encoder import GraphGRU_Encoder
from models.GCN import GMulitProp
from models.MLP import Mapping
from models.Propagator import PropagationNetwork
from tools.utils import chrono_embedding_sincos, kl_normal_log

class CausalKoopman(nn.Module):
    def __init__(self,
                 # For encoder an decoder
                 attr_dim, state_dim, control_dim, edge_dim, num_for_predict,
                 g_dim, node_hidden_dim, edge_hidden_dim, effect_hidden_dim, node_num,
                 residual=False, enc_type='pn', dec_type='pn', norm_type='Layer', gdep=2, prior_graph=None,
                 num_for_target=1, POI_vector=None, use_node_adapt=True, alpha=0.3,
                 random_sampling=True, causal_mask=False, use_Gprop=True, use_spatial_transformer=False,
                 node_regularize=False, aug_reweight=True, use_aug=True, use_pred_g=False, strict_invariant_err=True,

                 # For Koopman
                 regularize_rank=True, use_global_operator=True, use_community_operator=True, node_attention=True,
                 use_encoder_control=False, use_adaAdj=True, sample_num=10,
                 use_control=True,
                 community_num=10, community_dim=64,

                 # For prior
                 use_prior=True,
                 noise_dist_type='mlp',
                 base_dist_type='gaussian',
                 device='cuda:0'):
        super(CausalKoopman, self).__init__()


        self.residual = residual
        self.prior_graph = prior_graph
        self.sample_num = sample_num
        self.POI_vector = POI_vector
        self.attr_dim = attr_dim
        self.enc_type = enc_type
        self.dec_type = dec_type
        self.node_regularize = node_regularize
        self.use_spatial_transformer = use_spatial_transformer
        self.aug_reweight = aug_reweight
        self.use_aug = use_aug
        self.g_dim = g_dim
        self.num_for_target = num_for_target
        self.use_pred_g = use_pred_g
        self.strict_invariant_err = strict_invariant_err

        if self.POI_vector is not None:
            poi_dim = self.POI_vector.shape[1]
            self.POI_vector = torch.tensor(self.POI_vector).to(torch.float32).to(device)
            self.POI_vector = self.POI_vector/self.POI_vector.sum(dim=[1], keepdim=True)
            self.POI_vector = torch.where(torch.isnan(self.POI_vector),
                                          torch.full_like(self.POI_vector, 0),
                                          self.POI_vector)
            control_dim += poi_dim

        if norm_type == 'Layer':
            self.norm = nn.LayerNorm
        elif norm_type == 'Instance':
            self.norm = nn.InstanceNorm1d

        """########################################### Encoder ######################################################"""
        # 编码器 求g(x)
        # 编码器不输入控制变量
        self.use_control = use_control

        if self.use_control:
            self.encode_control = nn.Linear(control_dim, g_dim)
            control_hidden_dim = g_dim
        else:
            control_hidden_dim = 0
            control_dim = 0

        input_node_dim = attr_dim + state_dim
        input_node_dim += control_hidden_dim if use_encoder_control else 0
        input_edge_dim = state_dim + edge_dim
        self.use_Gprop = use_Gprop
        if self.use_Gprop:
            prior_graph_num = use_node_adapt + use_spatial_transformer + (self.prior_graph is not None)
        else:
            prior_graph_num = 1

        if self.use_Gprop:
            self.pror_function = GMulitProp(prior_graph_num)
        else:
            self.pror_function = None

        """########################################### Encoder ######################################################"""
        self.causal_mask = causal_mask
        self.fusion_node_embedding = nn.Linear(g_dim + control_hidden_dim, g_dim)
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=g_dim,
            nhead=1 if self.use_control else 4,
            dim_feedforward=effect_hidden_dim,
            batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(
            self.encoder_layer, num_layers=gdep, norm=self.norm(g_dim))

        if enc_type == 'GraphGRU':
            self.state_encoder = GraphGRU_Encoder(node=node_num,
                                                  num_for_predict=num_for_predict,
                                                  input_dim=input_node_dim,
                                                  latent_dim=g_dim,
                                                  hidden_dim=g_dim,
                                                  gcn_depth=gdep,
                                                  dropout_prob=0.3,
                                                  alpha=alpha,
                                                  encoder_type=enc_type,
                                                  use_control=use_encoder_control,
                                                  random_sampling=random_sampling,
                                                  prop_function=self.pror_function)
        self.dist_latent = Mapping(encoder_dim=g_dim, dist_dim=g_dim, hidden_dims=[32])

        """########################################### Decoder ######################################################"""
        input_node_dim = attr_dim + g_dim
        input_node_dim += control_hidden_dim if use_encoder_control else 0
        input_edge_dim = g_dim + edge_dim
        if dec_type == 'pn':
            self.state_decoder = PropagationNetwork(
                attr_dim, state_dim, control_hidden_dim, edge_dim, node_hidden_dim, edge_hidden_dim, effect_hidden_dim,
                input_node_dim=input_node_dim, input_edge_dim=input_edge_dim,
                output_dim=state_dim, use_control=use_encoder_control, tanh=False,
                residual=residual, prop_function=self.pror_function)

        """########################################## Detector #####################################################"""
        if use_community_operator:
            self.detector = CommunityDetector(node=node_num,
                                              time=num_for_predict,
                                              input_dim=state_dim,
                                              attr_dim=attr_dim,
                                              control_dim=self.POI_vector.shape[-1] if self.POI_vector is not None else 0,
                                              hidden_dim=effect_hidden_dim,
                                              community_num=community_num,
                                              community_embedding_dim=community_dim, )
            self.community = None

        """########################################## Prior Graph #####################################################"""
        self.use_adaAdj = use_adaAdj
        self.use_node_adapt = use_node_adapt
        self.node_attention = node_attention
        if self.use_node_adapt:
            self.nodevec1 = nn.Parameter(torch.randn(node_num, 32).to(device), requires_grad=True)
            self.nodevec2 = nn.Parameter(torch.randn(32, node_num).to(device), requires_grad=True)

        self.attention_spatial = nn.MultiheadAttention(
            embed_dim=num_for_predict * (attr_dim + state_dim + control_dim),
            num_heads=2 if self.use_control else 4,
            batch_first=True)

        """########################################## Dynamcial #####################################################"""

        self.community_num = community_num
        self.use_global_operator = use_global_operator
        self.use_community_operator = use_community_operator
        self.weight_koopman = nn.Parameter(torch.ones(3)/3, requires_grad=True)

        if use_global_operator:
            self.global_koopman_operator = nn.Parameter(torch.rand(g_dim, g_dim))
            nn.init.orthogonal_(self.global_koopman_operator)
            if self.use_control:
                self.global_control_operator = nn.Parameter(torch.rand(control_hidden_dim, g_dim))
                nn.init.orthogonal_(self.global_control_operator)

        if self.use_community_operator:
            self.community_koopman_operator = nn.Parameter(torch.rand(community_num, g_dim, g_dim))
            if self.use_control:
                self.community_control_operator = nn.Parameter(torch.rand(community_num, control_hidden_dim, g_dim))
            for c in range(community_num):
                nn.init.orthogonal_(self.community_koopman_operator[c])
                if self.use_control:
                    nn.init.orthogonal_(self.community_control_operator[c])

        self.causal_mask = causal_mask
        self.to_q = nn.Linear(g_dim, g_dim + control_hidden_dim)
        if self.node_attention:
            self.attention = nn.MultiheadAttention(
                embed_dim=num_for_predict,
                num_heads=1 if self.use_control else 4,
                batch_first=True)

        self.node_koopman_operator = None
        if self.use_control:
            self.node_control_operator = None
        self.controls_dim = control_hidden_dim

        if self.use_adaAdj:
            adaAdj_dim = (num_for_predict - 1) * state_dim
            self.adaAdj_operator = [nn.Linear(adaAdj_dim, effect_hidden_dim)]
            self.adaAdj_operator += [self.norm(effect_hidden_dim)]
            self.adaAdj_operator += [nn.ReLU(True)]
            for _ in range(gdep - 1):
                self.adaAdj_operator += [nn.Linear(effect_hidden_dim, effect_hidden_dim)]
                self.adaAdj_operator += [self.norm(effect_hidden_dim)]
                self.adaAdj_operator += [nn.ReLU(True)]
            self.adaAdj_operator += [nn.Linear(effect_hidden_dim, g_dim + control_hidden_dim)]
            self.adaAdj_operator = nn.Sequential(*self.adaAdj_operator)

        """######################################### Causal Prior ##################################################"""
        self.use_prior = use_prior
        if self.use_prior:
            self.causal_prior = CausalPrior(node=node_num,
                                            input_dim=input_node_dim,
                                            time=num_for_predict,
                                            latent_dim=g_dim,
                                            domain_num=community_num,
                                            domain_dim=community_dim,
                                            hidden_dim=g_dim,
                                            layer_num=gdep,
                                            noise_dist_type=noise_dist_type,
                                            base_dist_type=base_dist_type,
                                            device=device,
                                            pror_function=self.pror_function,
                                            tao=1, )

    def step(self, g, u=None, rel_attrs=None, node_koopman_operator=None, node_control_operator=None):
        aug_g = g
        if self.use_control:
            aug_u = u

        global_step = 0
        if self.use_global_operator:
            global_step = torch.matmul(aug_g, self.global_koopman_operator)
            if self.use_control:
                global_step += torch.matmul(aug_u, self.global_control_operator)

        # community
        community_step = 0
        if self.use_community_operator:
            for c in range(self.community_num):
                A = self.community_koopman_operator[c]
                if self.use_control:
                    B = self.community_control_operator[c]
                g_c = aug_g*self.community[:, c][None, :, None]
                if self.use_control:
                    u_c = aug_u*self.community[:, c][None, :, None]
                tmp = torch.matmul(g_c, A)
                if self.use_control:
                    tmp +=torch.matmul(u_c, B)
                community_step += tmp

        # node
        node_koopman_operator = self.node_koopman_operator
        if self.use_control:
            node_control_operator = self.node_control_operator
        node_step = torch.einsum('bnd, bndl-> bnl', (aug_g, node_koopman_operator))

        if self.use_control:
            node_step += torch.einsum('bnd, bndl-> bnl', (aug_u, node_control_operator))

        weight_koopman = F.softmax(self.weight_koopman, dim=-1)
        node_step = weight_koopman[0]*global_step + \
                    weight_koopman[1]*community_step + \
                    weight_koopman[2]*node_step   # + community_step

        return node_step, global_step, community_step, node_step

    def simulate(self, g, u_seq=None, T=None, rel_attrs=None, node_koopman_operator=None, node_control_operator=None):
        g_list = []
        global_list = []
        community_list = []
        node_list = []
        for t in range(T):
            if self.use_control:
                scale_step = self.step(g, u_seq[:, :, t], rel_attrs, node_koopman_operator, node_control_operator)
            else:
                scale_step = self.step(g, rel_attrs=rel_attrs, node_koopman_operator=node_koopman_operator)
            g = scale_step[0]
            g_list.append(g.unsqueeze(2))
            global_list.append(scale_step[1].unsqueeze(2))
            community_list.append(scale_step[2].unsqueeze(2))
            node_list.append(scale_step[3].unsqueeze(2))

        return torch.cat(g_list, 2), torch.cat(global_list, 2), torch.cat(community_list, 2), torch.cat(node_list, 2)

    def regularize_rank_loss(self, ):
        global_rank_regularizer = 0
        if self.use_global_operator:

            if self.use_control:
                global_rank_regularizer += 0.5 * torch.linalg.cond(self.global_koopman_operator)
                global_rank_regularizer += 0.5 * torch.linalg.cond(self.global_control_operator)
            else:
                global_rank_regularizer += torch.linalg.cond(self.global_koopman_operator)

            global_rank_regularizer -= 1

        community_rank_regularizer = 0
        if self.use_community_operator:

            if self.use_control:
                community_rank_regularizer += 0.5 * torch.linalg.cond(self.community_koopman_operator).mean()
                community_rank_regularizer += 0.5 * torch.linalg.cond(self.community_control_operator).mean()
            else:
                community_rank_regularizer += torch.linalg.cond(self.community_koopman_operator).mean()

            community_rank_regularizer -= 1

        node_rank_regularizer = 0
        if self.node_regularize:
            batch, node, dim, _ = self.node_koopman_operator.shape
            node_koopman = self.node_koopman_operator.reshape(batch * node, dim, dim)
            mat1 = torch.bmm(node_koopman, node_koopman.transpose(1, 2))
            diversity_loss1 = (mat1 - torch.eye(mat1.shape[1]).to(mat1.device))
            if self.use_control:
                node_control = self.node_control_operator.reshape(batch * node, dim, dim)
                mat2 = torch.bmm(node_control, node_control.transpose(1, 2))
                diversity_loss2 = (mat2 - torch.eye(mat2.shape[1]).to(mat2.device))
                node_rank_regularizer = 0.5*torch.norm(diversity_loss1, p=2, dim=[-1,-2]).mean()+\
                                        0.5*torch.norm(diversity_loss2, p=2, dim=[-1,-2]).mean()
            else:
                node_rank_regularizer = torch.norm(diversity_loss1, p=2, dim=[-1, -2]).mean()

        return global_rank_regularizer + community_rank_regularizer + node_rank_regularizer


    def eig_loss_component(self, Ct):
        D = torch.linalg.eigvals(Ct)
        eig_loss = D.abs() - torch.ones_like(D, device=D.device)
        eig_loss = torch.norm(eig_loss, p=2).mean()

        return eig_loss


    def eig_loss(self):

        loss = 0
        global_loss = 0
        if self.use_control:
            global_loss += (0.5*self.eig_loss_component(self.global_koopman_operator)+
                            0.5*self.eig_loss_component(self.global_control_operator))
        else:
            global_loss += self.eig_loss_component(self.global_koopman_operator)

        loss += global_loss

        community_loss = 0
        for c in range(self.community_num):
            if self.use_control:
                community_loss += (0.5*self.eig_loss_component(self.community_koopman_operator[c])+
                                   0.5*self.eig_loss_component(self.community_control_operator[c]))
            else:
                community_loss += self.eig_loss_component(self.community_koopman_operator[c])

        community_loss /= self.community_num

        loss += community_loss

        # node_loss = 0
        # if self.node_regularize:
        #     node_koopman_operator = self.node_koopman_operator.mean(0)
        #     node_control_operator = self.node_control_operator.mean(0)
        #     for n in range(len(node_koopman_operator)):
        #         node_loss += (0.5*self.eig_loss_component(node_koopman_operator[n]) +
        #                       0.5*self.eig_loss_component(node_control_operator[n]))
        #     node_loss /= len(node_koopman_operator)
        #
        #     loss += node_loss

        return loss

    def forward(self, attrs, states, controls=None):
        batch, node, time = states.shape[:3]
        if self.use_control:
            controls = controls.unsqueeze(1).expand([-1, node, -1, -1])
            if self.POI_vector is not None:
                POI_vector = self.POI_vector.unsqueeze(dim=0).unsqueeze(dim=2).expand(batch, -1, controls.shape[2], -1)
                controls_origin = torch.cat([controls, POI_vector], dim=-1)
            else:
                POI_vector = None
                controls_origin = controls
            controls = self.encode_control(controls_origin)

        # 0 dayofweek   1 hourofday      2 holiday
        if time == 12:
            t_emb = chrono_embedding_sincos(attrs[..., :2], [7, 288])
        else:
            t_emb = chrono_embedding_sincos(attrs[..., :2], [7, 48])
        attrs = t_emb.unsqueeze(1).expand([-1, node, -1, -1])
        if self.use_community_operator:
            if self.use_control:
                self.community = self.detector(states, POI_vector[:,:,:time])            # N*C
            else:
                self.community = self.detector(states)

        if self.use_prior:
            base_noise_dist = D.MultivariateNormal(torch.zeros(self.g_dim, device=states.device),
                                 torch.eye(self.g_dim, device=states.device))

        prior_graph = []
        if self.prior_graph is not None:
            prior_graph += [self.prior_graph]

        if self.use_spatial_transformer:
            if self.use_control:
                spatial_input = torch.cat([states, attrs[:, :, :time], controls_origin[:, :, :time]], dim=-1).reshape(batch, node, -1)
            else:
                spatial_input = torch.cat([states, attrs[:, :, :time]], dim=-1).reshape(batch, node, -1)
            prior_graph += [self.attention_spatial(spatial_input,
                                                   spatial_input,
                                                   spatial_input)[1]]
        if self.use_node_adapt:
            prior_graph += [F.softmax(F.relu(torch.mm(self.nodevec1, self.nodevec2)), dim=-1)]

        if self.use_control:
            node_embedding, hidden_states = self.state_encoder(attrs[:,:,:time], states, controls[:,:,:time], prior_graph)
            node_embedding = self.fusion_node_embedding(torch.cat([node_embedding, controls[:, :, :time]], dim=-1))
        else:
            node_embedding, hidden_states = self.state_encoder(attrs[:, :, :time], states, rel_attrs=prior_graph)
            node_embedding = self.fusion_node_embedding(node_embedding)

        node_embedding = node_embedding.reshape(batch*node, time, -1)
        if self.causal_mask:
            mask = nn.Transformer.generate_square_subsequent_mask(time).to(node_embedding.device)
            tmp = self.transformer_encoder(node_embedding, is_causal=True, mask=mask)
        else:
            tmp = self.transformer_encoder(node_embedding)

        tmp = tmp.reshape(batch, node, time, -1)

        q_mus, q_logvars, q_zx = self.dist_latent(tmp, self.use_prior)

        gx_vec = q_zx.rsample()

        attrs_for_recon = attrs[:, :, :time]
        attrs_for_hispred = attrs[:, :, 1:time]
        attrs_for_pred = attrs[:, :, time:]
        if self.use_control:
            controls_for_recon = controls[:, :, :time]
            controls_for_hispred = controls[:, :, 1:time]
            controls_for_pred = controls[:, :, time:]

        if self.dec_type == 'pn':
            if self.use_control:
                reconstructions, _ = self.state_decoder(attrs=attrs_for_recon, states=gx_vec,
                                                        controls=controls_for_recon, rel_attrs=prior_graph)
            else:
                reconstructions, _ = self.state_decoder(attrs=attrs_for_recon, states=gx_vec, rel_attrs=prior_graph)

        node_fit_err = 0
        """ generate node koopman based on input data """
        if self.node_attention:
            trans_out = gx_vec.reshape(batch*node, time, -1)
            q_trans_out = self.to_q(trans_out.reshape(batch*node, time, -1))
            local_transform = self.attention(q_trans_out.transpose(1,2),
                                             trans_out.transpose(1,2),
                                             trans_out.transpose(1,2))[1]
            node_koopman_operator = local_transform[:, :self.g_dim].reshape(batch, node, self.g_dim, self.g_dim)
            if self.use_control:
                node_control_operator = local_transform[:, self.g_dim:].reshape(batch, node, self.controls_dim, self.g_dim)

        self.node_koopman_operator = node_koopman_operator
        if self.use_control:
            self.node_control_operator = node_control_operator

        """
        rollout 0 -> 1 : T + 1  
        """
        T = time - 1
        if self.use_control:
            U_for_hispred = controls[:, :, :T]
            set_for_hispred = self.simulate(T=T, g=gx_vec[:, :, 0], u_seq=U_for_hispred, rel_attrs=prior_graph,
                                            node_koopman_operator=self.node_koopman_operator,
                                            node_control_operator=self.node_control_operator)
        else:
            set_for_hispred = self.simulate(T=T, g=gx_vec[:, :, 0], rel_attrs=prior_graph,
                                            node_koopman_operator=self.node_koopman_operator)

        G_for_hispred, global_for_hispred, community_for_hispred, node_for_hispred = set_for_hispred

        if self.dec_type=='pn':
            if self.use_control:
                history_pred, _ = self.state_decoder(attrs=attrs_for_hispred, states=G_for_hispred,
                                                     controls=controls_for_hispred, rel_attrs=prior_graph, )
            else:
                history_pred, _ = self.state_decoder(attrs=attrs_for_hispred,
                                                     states=G_for_hispred,
                                                     rel_attrs=prior_graph, )
        """
        space loss 
        """
        invariant_err = gx_vec[:, :, 1:time] - G_for_hispred

        if self.strict_invariant_err:
            invariant_err = (invariant_err ** 2).sum(dim=[-1, -2]).mean()
        else:
            invariant_err = (invariant_err ** 2).mean()

        if node_fit_err == 0:
            node_fit_err = invariant_err
            loss_metric = invariant_err

        """
        compute adaAdj
        """
        if self.use_adaAdj:
            pred_diff = history_pred.reshape(batch*self.sample_num, node, -1) - states[:, :, 1:].reshape(batch*self.sample_num, node, -1)
            adaAdj = self.adaAdj_operator(pred_diff)
            adaAdj_A = adaAdj[..., :self.g_dim]
            adaAdj_A = torch.diag_embed(adaAdj_A)  # (B,D, D)
            self.node_koopman_operator = self.node_koopman_operator + adaAdj_A

            if self.use_control:
                adaAdj_B = adaAdj[..., self.g_dim:]
                if node_control_operator.shape[-1] == adaAdj_B.shape[-1]:
                    adaAdj_B = torch.diag_embed(adaAdj_B)  # (B,D, D)
                    self.node_control_operator = self.node_control_operator + adaAdj_B
                else:
                    self.node_control_operator = self.node_control_operator + adaAdj_B[..., None]

        """
        predict for future
        """
        if self.use_control:
            U_for_pred = controls[:, :, T:]

        if self.use_pred_g:
            if self.use_control:
                set_for_pred = self.simulate(T=self.num_for_target, g=G_for_hispred[:, :, -1], u_seq=U_for_pred,
                                             rel_attrs=prior_graph,
                                             node_koopman_operator=self.node_koopman_operator,
                                             node_control_operator=self.node_control_operator)
            else:
                set_for_pred = self.simulate(T=self.num_for_target, g=G_for_hispred[:, :, -1],
                                             rel_attrs=prior_graph,
                                             node_koopman_operator=self.node_koopman_operator,)

        else:
            if self.use_control:
                set_for_pred = self.simulate(T=self.num_for_target, g=gx_vec[:, :, -1], u_seq=U_for_pred,
                                             rel_attrs=prior_graph,
                                             node_koopman_operator=self.node_koopman_operator,
                                             node_control_operator=self.node_control_operator)
            else:
                set_for_pred = self.simulate(T=self.num_for_target, g=gx_vec[:, :, -1],
                                             rel_attrs=prior_graph,
                                             node_koopman_operator=self.node_koopman_operator,)

        G_for_pred, global_for_pred, community_for_pred, node_for_pred = set_for_pred

        if self.dec_type=='pn':
            if self.use_control:
                future_pred, _ = self.state_decoder(attrs=attrs_for_pred, states=G_for_pred,
                                                    controls=controls_for_pred, rel_attrs=prior_graph)
            else:
                future_pred, _ = self.state_decoder(attrs=attrs_for_pred, states=G_for_pred, rel_attrs=prior_graph)


        if self.use_prior:
            """
            prior causal constrained
            """
            log_pz_global = self.causal_prior(gx_vec, self.global_koopman_operator, prior_graph=prior_graph,
                                              community=None,
                                              scale_type='global',
                                              noise_dist=base_noise_dist)
            log_pz_community = self.causal_prior(gx_vec, self.community_koopman_operator, prior_graph=prior_graph,
                                                 community=self.community,
                                                 scale_type='community',
                                                 noise_dist=base_noise_dist)
            log_pz_node = self.causal_prior(gx_vec, self.node_koopman_operator, prior_graph=prior_graph,
                                                 community=None,
                                                 scale_type='node',
                                                 noise_dist=base_noise_dist)
            log_pz = (log_pz_global + log_pz_community + log_pz_node)/3.0

            # 后验
            log_qzx = q_zx.log_prob(gx_vec)
            IM = (log_qzx - log_pz).mean()
        else:
            IM = kl_normal_log(q_mus, q_logvars, torch.zeros_like(q_mus).to(q_mus.device),
                           torch.zeros_like(q_logvars))

        return future_pred, history_pred, reconstructions, node_fit_err, invariant_err, loss_metric, IM