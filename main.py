#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import argparse
import copy
from datetime import datetime
import torch
# from torch.utils.tensorboard import SummaryWriter
import pandas as pd
import numpy as np
import time
import sys
import os
from models.CausalKoopman import CausalKoopman
from config.config import get_logger
from tools.utils import sym_adj, sym_adj, asym_adj
import yaml

try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper

if __name__ == '__main__':

    data_name = 'NYC2016'
    config_filename = 'config/config_{}.yaml'.format(data_name)
    with open(config_filename, 'r') as ymlfile:
        cfg = yaml.load(ymlfile, Loader=Loader)

    torch.manual_seed(1234)
    base_path = cfg['base_path']

    dataset_name = cfg['dataset_name']
    dataset_path = os.path.join(base_path, dataset_name)

    if cfg['data']['name'] == 'BJ2021' or cfg['data']['name'] == 'BJ2022':
        from dataset.datasets_BJ import load_dataset
        from ModelTest_BJ import baseline_test
        from ModelTrain_BJ import baseline_train
        graph_name ='geo_adj.npy'
    elif cfg['data']['name'] == 'NYC':
        from dataset.datasets_NYC import load_dataset
        from ModelTest_NYC import baseline_test
        from ModelTrain_NYC import baseline_train
        graph_name = 'geo_adj.npy'
    elif cfg['data']['name'] == 'PEMS04':
        from dataset.datasets_PEMS import load_dataset
        from ModelTest_PEMS import baseline_test
        from ModelTrain_PEMS import baseline_train
        graph_name = '{:s}_adj.npy'.format(cfg['data']['name'])
    elif cfg['data']['name'] == 'PEMS08':
        from dataset.datasets_PEMS import load_dataset
        from ModelTest_PEMS import baseline_test
        from ModelTrain_PEMS import baseline_train
        graph_name = '{:s}_adj.npy'.format(cfg['data']['name'])

    log_path = os.path.join('Results', cfg['data']['name'], cfg['model_name'],  'exp{:s}'.format(cfg['expid']), 'log')
    if not os.path.exists(log_path):
        os.makedirs(log_path, exist_ok=True)

    save_path = os.path.join('Results', cfg['data']['name'], cfg['model_name'],  'exp{:s}'.format(cfg['expid']), 'ckpt')
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)


    log_dir = log_path
    log_level = 'INFO'
    log_name = 'info_' + datetime.now().strftime('%m-%d_%H:%M') + '.log'
    logger = get_logger(log_dir, __name__, log_name, level=log_level, write_to_file=not cfg['test_only'])

    if not cfg['test_only']:
        confi_name = 'config{:s}_'.format(cfg['expid']) + datetime.now().strftime('%m-%d_%H:%M') + '.yaml'
        with open(os.path.join(log_dir, confi_name), 'w+') as _f:
            yaml.safe_dump(cfg, _f)

    logger.info(cfg)
    logger.info(dataset_path)
    logger.info(log_path)

    writer = None
    if cfg['train']['tensorboard']:
        tensorboard_path = os.path.join('Results', cfg['model_name'], 'exp{:s}'.format(cfg['expid']), 'tensorboard_log')
        if not os.path.exists(tensorboard_path):
            os.makedirs(tensorboard_path, exist_ok=True)
        writer = SummaryWriter(tensorboard_path)
        logger.info(tensorboard_path)

    device = torch.device(cfg['device'])

    dataloader = load_dataset(dataset_path,
                              cfg['data']['train_batch_size'],
                              cfg['data']['val_batch_size'],
                              cfg['data']['test_batch_size'],
                              logger=logger, device=device,
                              )

    try:
        pre_graph = None
        if cfg['model']['adj'] == 'adj':
            pre_graph = np.load(os.path.join(base_path, 'graph', graph_name)).astype(np.float32)

        elif cfg['model']['adj'] == 'affinity':
            pre_graph = np.load(os.path.join(base_path, 'graph', 'geo_affinity.npy')).astype(np.float32)

        if cfg['model']['norm_graph'] == 'sym':
            static_norm_adjs = torch.tensor(sym_adj(pre_graph)).to(device)
        elif cfg['model']['norm_graph'] == 'asym':
            static_norm_adjs = torch.tensor(asym_adj(pre_graph)).to(device)
        else:
            static_norm_adjs = torch.tensor(pre_graph).to(device)
    except:
        static_norm_adjs = None

    try:
        if cfg['data']['external']:
            external = pd.read_hdf(os.path.join(base_path, 'poi.h5')).values
    except:
        external = None

    model_name = cfg['model_name']

    val_loss_list = []
    val_mae_list = []
    val_mape_list = []
    val_rmse_list = []

    test_loss_list = []
    test_mae_list = []
    test_mape_list = []
    test_rmse_list = []
    for runid in range(cfg['runs']):
        if cfg['model_name']=='CausalKoopman':
            model = CausalKoopman(  # global
                # For encoder an decoder
                attr_dim=cfg['model']['attr_dim'],
                state_dim=cfg['model']['state_dim'],
                control_dim=cfg['model']['control_dim'],
                POI_vector=external,
                edge_dim=cfg['model']['edge_dim'],
                num_for_predict=cfg['data']['num_for_predict'],
                g_dim=cfg['model']['g_dim'],
                node_hidden_dim=cfg['model']['node_hidden_dim'],
                edge_hidden_dim=cfg['model']['edge_hidden_dim'],
                effect_hidden_dim=cfg['model']['effect_hidden_dim'],
                node_num=cfg['model']['node_num'],
                residual=cfg['model']['residual'],
                enc_type=cfg['model']['enc_type'],
                dec_type=cfg['model']['dec_type'],
                norm_type=cfg['model']['norm_type'],
                gdep=cfg['model']['gdep'],
                alpha=cfg['model']['alpha'],
                prior_graph=static_norm_adjs,
                num_for_target=cfg['data']['num_for_target'],
                use_node_adapt=cfg['model']['use_node_adapt'],
                causal_mask=cfg['model']['causal_mask'],
                use_Gprop=cfg['model']['use_Gprop'],
                node_regularize=cfg['model']['node_regularize'],
                aug_reweight=cfg['model']['aug_reweight'],
                use_aug=cfg['model']['use_aug'],
                use_pred_g=cfg['model']['use_pred_g'],
                strict_invariant_err=cfg['model']['strict_invariant_err'],

                # For Koopman
                use_prior=cfg['model']['use_prior'],
                regularize_rank=cfg['train']['regularize_rank'],
                use_global_operator=cfg['model']['use_global_operator'],
                use_community_operator=cfg['model']['use_community_operator'],
                node_attention=cfg['model']['node_attention'],
                use_spatial_transformer=cfg['model']['use_spatial_transformer'],
                use_encoder_control=cfg['model']['use_encoder_control'],
                use_control=cfg['model']['use_control'],
                use_adaAdj=cfg['model']['use_adaAdj'],
                sample_num=cfg['model']['sample_num'],
                community_num=cfg['model']['community_num'],
                community_dim=cfg['model']['community_dim'],
                device=device)


        logger.info(model_name)

        if cfg['test_only']:
            mvalid_total_loss, mvalid_pred_mae, mvalid_pred_rmse, mvalid_pred_mape, \
            mtest_total_loss, mtest_pred_mae, mtest_pred_rmse, mtest_pred_mape = baseline_test(runid,
                                                                                               model,
                                                                                               dataloader,
                                                                                               device,
                                                                                               logger,
                                                                                               cfg,
                                                                                               writer,
                                                                                               )
        else:
            mvalid_total_loss, mvalid_pred_mae, mvalid_pred_rmse, mvalid_pred_mape, \
            mtest_total_loss, mtest_pred_mae, mtest_pred_rmse, mtest_pred_mape = baseline_train(runid,
                                                                                                model,
                                                                                                dataloader,
                                                                                                device,
                                                                                                logger,
                                                                                                cfg,
                                                                                                writer, )
        val_loss_list.append(mvalid_total_loss)
        val_mae_list.append(mvalid_pred_mae)
        val_mape_list.append(mvalid_pred_mape)
        val_rmse_list.append(mvalid_pred_rmse)

        test_loss_list.append(mtest_total_loss)
        test_mae_list.append(mtest_pred_mae)
        test_mape_list.append(mtest_pred_mape)
        test_rmse_list.append(mtest_pred_rmse)

    test_loss_list = np.array(test_loss_list)
    test_mae_list = np.array(test_mae_list)
    test_mape_list = np.array(test_mape_list)
    test_rmse_list = np.array(test_rmse_list)

    aloss = np.mean(test_loss_list, 0)
    amae = np.mean(test_mae_list, 0)
    amape = np.mean(test_mape_list, 0)
    armse = np.mean(test_rmse_list, 0)

    sloss = np.std(test_loss_list, 0)
    smae = np.std(test_mae_list, 0)
    smape = np.std(test_mape_list, 0)
    srmse = np.std(test_rmse_list, 0)

    logger.info('valid\tLoss\tMAE\tRMSE\tMAPE')
    log = 'mean:\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}'
    logger.info(
        log.format(np.mean(val_loss_list), np.mean(val_mae_list), np.mean(val_rmse_list), np.mean(val_mape_list)))
    log = 'std:\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}'
    logger.info(log.format(np.std(val_loss_list), np.std(val_mae_list), np.std(val_rmse_list), np.std(val_mape_list)))
    logger.info('\n\n')

    logger.info('Test\tLoss\tMAE\tRMSE\tMAPE')
    log = 'mean:\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}'
    logger.info(
        log.format(np.mean(test_loss_list), np.mean(test_mae_list), np.mean(test_rmse_list), np.mean(test_mape_list)))
    log = 'std:\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}'
    logger.info(
        log.format(np.std(test_loss_list), np.std(test_mae_list), np.std(test_rmse_list), np.mean(test_mape_list)))
    logger.info('\n\n')
