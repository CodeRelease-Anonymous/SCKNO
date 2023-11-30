#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
from torch.cuda.amp import autocast
from torch.nn.parallel import DistributedDataParallel

from tools.metrics import masked_mae_torch, masked_mape_torch, masked_rmse_torch, metric, metric_all
from tools.utils import StepLR2

class Trainer():
    def __init__(self,
                 model,
                 base_lr,
                 weight_decay,
                 milestones,
                 lr_decay_ratio,
                 min_learning_rate,
                 max_grad_norm,
                 num_for_target,
                 num_for_predict,
                 scaler,
                 device,
                 loss_weight,
                 rec=True,
                 pred=True,
                 hispred= True,
                 regularize_rank=True,
                 orthogonal=True,
                 sparse=True,
                 eig_rank=False,
                 ):
        self.scaler = scaler
        self.model = model

        self.device = device
        self.max_grad_norm = max_grad_norm
        self.loss_weight = loss_weight
        self.rec = rec
        self.pred = pred
        self.regularize_rank = regularize_rank
        self.orthogonal = orthogonal
        self.sparse = sparse
        self.eig_rank = eig_rank
        self.hispred= hispred
        self.model = model.to(device)

        self.optimizer = optim.Adam(self.model.parameters(), lr=base_lr, weight_decay=weight_decay)

        self.scheduler = StepLR2(optimizer=self.optimizer,
                                 milestones=milestones,
                                 gamma=lr_decay_ratio,
                                 min_lr=min_learning_rate)

        self.SmoothL1loss = nn.SmoothL1Loss(reduction='mean')
        self.scaler = scaler
        self.num_for_target = num_for_target
        self.num_for_predict = num_for_predict

    def train(self, x, pos, external=None, kl_weight=1):
        """
        """
        input_x = x[:, :, :self.num_for_predict]
        batch, node, time, input_dim = input_x.shape

        self.model.train()
        self.optimizer.zero_grad()
        future_pred, history_pred, reconstructions, \
        node_fit_err, invariant_err, loss_metric, IM = self.model(attrs=pos, states=self.scaler.transform(input_x),
                                                                  controls=external)

        IM = IM*kl_weight

        total_loss, \
        pred_loss, hispred_loss, rec_loss, \
        invariant_err, node_fit_err, loss_metric, \
        L1_loss, IM, \
        rec_mae, rec_rmse, rec_mape, \
        hispred_mae, hispred_rmse, hispred_mape, \
        pred_mae, pred_rmse, pred_mape, \
        rec_output, hispred_output, pred_output = self.get_loss(x, future_pred, history_pred, reconstructions,
                                                node_fit_err, invariant_err, loss_metric, IM)

        total_loss.backward()

        if self.max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

        self.optimizer.step()

        return total_loss.item(), \
                pred_loss.item(), hispred_loss.item(), rec_loss.item(), \
                invariant_err.item(), node_fit_err.item(), loss_metric.item(), \
                L1_loss.item(), IM.item(), \
                rec_mae, rec_rmse, rec_mape, \
                hispred_mae, hispred_rmse, hispred_mape, \
                pred_mae, pred_rmse, pred_mape, \
                rec_output, hispred_output, pred_output

    def eval(self, x, pos, external=None):
        input_x = x[:, :, :self.num_for_predict]
        batch, node, time, input_dim = input_x.shape

        self.model.eval()
        with torch.no_grad():
            future_pred, history_pred, reconstructions, \
            node_fit_err, invariant_err, loss_metric, IM = self.model(attrs=pos, states=self.scaler.transform(input_x),
                                                                      controls=external,)

            total_loss, \
            pred_loss, hispred_loss, rec_loss, \
            invariant_err, node_fit_err, loss_metric, \
            L1_loss, IM, \
            rec_mae, rec_rmse, rec_mape, \
            hispred_mae, hispred_rmse, hispred_mape, \
            pred_mae, pred_rmse, pred_mape, \
            rec_output, hispred_output, pred_output = self.get_loss(x, future_pred, history_pred, reconstructions,
                                                                    node_fit_err, invariant_err, loss_metric, IM)


        return [total_loss.item(),
                pred_loss.item(), hispred_loss.item(), rec_loss.item(),
                invariant_err.item(), node_fit_err.item(), loss_metric.item(),
                L1_loss.item(), IM.item(),
                rec_mae, rec_rmse, rec_mape,
                hispred_mae, hispred_rmse, hispred_mape,
                pred_mae, pred_rmse, pred_mape,
                rec_output, hispred_output, pred_output]

    def get_loss(self, x, future_pred, history_pred, reconstructions, node_fit_err, invariant_err, loss_metric, IM):

        input_x = x[:, :, :self.num_for_predict]
        batch, node, time, input_dim = input_x.shape

        ####################################### reconstruction loss  #######################################
        rec_loss = 0
        rec_mae, rec_rmse, rec_mape = 0, 0, 0
        rec_output = torch.zeros_like(x[:, :, :self.num_for_predict])
        if self.rec:
            rec_x = self.scaler.inverse_transform(reconstructions)
            rec_loss = self.SmoothL1loss(rec_x, input_x)
            rec_mae, rec_rmse, rec_mape = metric_all(
                [rec_x[..., 0:2], rec_x[..., 2:4],],
                [x[:, :, :self.num_for_predict, 0:2],
                 x[:, :, :self.num_for_predict, 2:4],], mode='NYC')

            rec_output = rec_x

        ####################################### prediction in history  #######################################
        hispred_loss = 0
        hispred_mae, hispred_rmse, hispred_mape = 0, 0, 0
        hispred_output = torch.zeros_like(x[:, :, 1:self.num_for_predict])
        if self.hispred:
            hispred_x = self.scaler.inverse_transform(history_pred)
            hispred_loss = self.SmoothL1loss(hispred_x, x[:, :, 1:self.num_for_predict])
            hispred_mae, hispred_rmse, hispred_mape = metric_all(
                [hispred_x[..., 0:2], hispred_x[..., 2:4],],
                [x[:, :, 1:self.num_for_predict, 0:2],
                 x[:, :, 1:self.num_for_predict, 2:4],], mode='NYC')

            hispred_output = hispred_x

        ####################################### prediction for future  #######################################
        pred_loss = 0
        pred_mae, pred_rmse, pred_mape = 0, 0, 0
        pred_output = torch.zeros_like(x[:, :, -1:])
        if self.pred:
            pred_x = self.scaler.inverse_transform(future_pred)

            pred_loss = self.SmoothL1loss(pred_x, x[:, :, -self.num_for_target:])
            pred_mae, pred_rmse, pred_mape = metric_all(
                    [pred_x[:, :, -self.num_for_target:, 0:2],
                     pred_x[:, :, -self.num_for_target:, 2:4],],
                    [x[:, :, -self.num_for_target:, 0:2],
                     x[:, :, -self.num_for_target:, 2:4],], mode='NYC')

            pred_output = pred_x

        ####################################### L1 loss  #######################################

        L1_loss = 0
        if self.regularize_rank:
            L1_loss += self.model.regularize_rank_loss()
        if self.orthogonal and self.model.use_community_operator:
            L1_loss += self.model.detector.embedding_constrains()
        if self.eig_rank:
            L1_loss += self.model.eig_loss()

        loss_weight = np.array(self.loss_weight)
        total_loss = loss_weight[0]*pred_loss + loss_weight[1]*hispred_loss + loss_weight[2]*rec_loss + \
                     loss_weight[3]*invariant_err+loss_weight[4]*node_fit_err + loss_weight[5]*loss_metric + \
                     loss_weight[6]*L1_loss+loss_weight[7]*IM


        return [total_loss,
               pred_loss, hispred_loss, rec_loss,
               invariant_err, node_fit_err, loss_metric,
               L1_loss, IM,
               rec_mae, rec_rmse, rec_mape,
               hispred_mae, hispred_rmse, hispred_mape,
               pred_mae, pred_rmse, pred_mape,
               rec_output,hispred_output, pred_output]
