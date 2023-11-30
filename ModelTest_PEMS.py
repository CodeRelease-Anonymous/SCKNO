#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import torch
import numpy as np
from tqdm import tqdm
from helper_PEMS import Trainer
import os
import torch.nn.functional as F
from tools.metrics import record, metric


def model_val(runid, engine, dataloader, device, logger, cfg, epoch, loss_list, data_dim_list):
    logger.info('Start validation phase.....')

    val_dataloder = dataloader['val']

    valid_loss_name = 'vaild_{:s}_loss'
    num_for_predict = cfg['data']['num_for_predict']
    num_for_target = cfg['data']['num_for_target']
    valid_dict_loss = {}
    for name in loss_list:
        valid_dict_loss[valid_loss_name.format(name)] = []

    valid_rec_mape = {}
    valid_rec_rmse = {}
    valid_rec_mae = {}

    valid_pred_mape = {}
    valid_pred_rmse = {}
    valid_pred_mae = {}

    valid_hispred_mape = {}
    valid_hispred_rmse = {}
    valid_hispred_mae = {}

    valid_total_list = []

    for name in data_dim_list:
        valid_rec_mae[name] = []
        valid_rec_rmse[name] = []
        valid_rec_mape[name] = []
        valid_pred_mae[name] = []
        valid_pred_rmse[name] = []
        valid_pred_mape[name] = []
        valid_hispred_mae[name] = []
        valid_hispred_rmse[name] = []
        valid_hispred_mape[name] = []

    val_tqdm_loader = tqdm(enumerate(val_dataloder))
    for iter, (x, pos) in val_tqdm_loader:
        # if iter>3:
        #     break
        x = x.to(device)
        pos = pos[:, 0, :, :]
        tpos = pos.to(device)
        if cfg['model']['use_control']:
            weather = torch.zeros((pos.shape[0], pos.shape[1], 1), device=device)
        else:
            weather = None

        output_list = engine.eval(x, tpos, weather)

        rec_set = output_list[len(loss_list):len(loss_list) + 3]
        hispred_set = output_list[len(loss_list) + 3:len(loss_list) + 6]
        pred_set = output_list[len(loss_list) + 6:len(loss_list) + 9]
        rec_output, hispred_output, pred_output = output_list[-3:]

        for i, key in enumerate(valid_dict_loss.keys()):
            valid_dict_loss[key].append(output_list[i])

        record(valid_rec_mae, valid_rec_rmse, valid_rec_mape, rec_set[0], rec_set[1], rec_set[2], mode="PeMS")
        record(valid_pred_mae, valid_pred_rmse, valid_pred_mape, pred_set[0], pred_set[1], pred_set[2], mode="PeMS")
        record(valid_hispred_mae, valid_hispred_rmse, valid_hispred_mape, hispred_set[0], hispred_set[1],
               hispred_set[2], mode="PeMS")

        valid_total_list.append(metric(pred_output[:, :, -num_for_target:], x[:, :, -num_for_target:]))

    mvalid_loss = []
    for i, key in enumerate(valid_dict_loss.keys()):
        mvalid_loss.append(np.mean(valid_dict_loss[key]))

    valid_total_list = np.array(valid_total_list)
    mvalid_total_list = np.mean(valid_total_list, axis=0)

    log = 'Epoch: {:03d}, Vaild Total Loss: {:.4f} Learning rate: {}\n' \
          'Vaild Pred Loss: {:.4f}\t\t\tVaild Hispred Loss: {:.4f}\n' \
          'Vaild Recon Loss: {:.4f}\t\t\tVaild Invariant Loss: {:.4f} \n' \
          'Vaild Nodefit Loss: {:.4f}\t\t\tVaild Metric Loss: {:.4f} \n' \
          'Vaild L1 Loss: {:.4f}\t\t\tVaild KL Loss: {:.4f} \n' \
          'Vaild Rec  Speed  MAE: {:.4f}\t\t\tRMSE: {:.4f}\t\t\tMAPE: {:.4f} \n' \
          'Vaild Hispred Speed  MAE: {:.4f}\t\t\tRMSE: {:.4f}\t\t\tMAPE: {:.4f}\n' \
          'Vaild Pred Speed  MAE: {:.4f}\t\t\tRMSE: {:.4f}\t\t\tMAPE: {:.4f}\n' \
          'Vaild Pred Total MAE: {:.4f}\t\t\tRMSE: {:.4f}\t\t\tMAPE: {:.4f}\n'
    logger.info(log.format(epoch, mvalid_loss[0], str(engine.scheduler.get_lr()),
                           mvalid_loss[1], mvalid_loss[2],
                           mvalid_loss[3], mvalid_loss[4],
                           mvalid_loss[5], mvalid_loss[6],
                           mvalid_loss[7], mvalid_loss[8],

                           np.mean(valid_rec_mae['speed']), np.mean(valid_rec_rmse['speed']),
                           np.mean(valid_rec_mape['speed']),

                           np.mean(valid_hispred_mae['speed']), np.mean(valid_hispred_rmse['speed']),
                           np.mean(valid_hispred_mape['speed']),

                           np.mean(valid_pred_mae['speed']), np.mean(valid_pred_rmse['speed']),
                           np.mean(valid_pred_mape['speed']),


                           mvalid_total_list[0], mvalid_total_list[2], mvalid_total_list[1]
                           ))

    return mvalid_loss[1], mvalid_total_list[0], mvalid_total_list[2], mvalid_total_list[1],


def model_test(runid, engine, dataloader, device, logger, cfg, loss_list, data_dim_list, mode='Test', ):
    logger.info('Start testing phase.....')

    test_dataloder = dataloader['test']

    test_loss_name = 'test_{:s}_loss'
    num_for_predict = cfg['data']['num_for_predict']
    num_for_target = cfg['data']['num_for_target']

    test_dict_loss = {}
    for name in loss_list:
        test_dict_loss[test_loss_name.format(name)] = []

    test_rec_mape = {}
    test_rec_rmse = {}
    test_rec_mae = {}

    test_pred_mape = {}
    test_pred_rmse = {}
    test_pred_mae = {}

    test_hispred_mape = {}
    test_hispred_rmse = {}
    test_hispred_mae = {}

    for name in data_dim_list:
        test_rec_mae[name] = []
        test_rec_rmse[name] = []
        test_rec_mape[name] = []
        test_pred_mae[name] = []
        test_pred_rmse[name] = []
        test_pred_mape[name] = []
        test_hispred_mae[name] = []
        test_hispred_rmse[name] = []
        test_hispred_mape[name] = []

    test_outputs_list = []
    test_targets_list = []
    test_metrics_list = []

    test_tqdm_loader = tqdm(enumerate(test_dataloder))
    for iter, (x, pos) in test_tqdm_loader:
        # if iter>3:
        #     break

        x = x.to(device)
        pos = pos[:, 0, :, :]
        tpos = pos.to(device)
        if cfg['model']['use_control']:
            weather = torch.zeros((pos.shape[0], pos.shape[1], 1), device=device)
        else:
            weather = None

        output_list = engine.eval(x, tpos, weather)

        rec_set = output_list[len(loss_list):len(loss_list) + 3]
        hispred_set = output_list[len(loss_list) + 3:len(loss_list) + 6]
        pred_set = output_list[len(loss_list) + 6:len(loss_list) + 9]
        rec_output, hispred_output, pred_output = output_list[-3:]

        test_outputs_list.append(pred_output)
        test_targets_list.append(x[:,:,-num_for_target:])
        test_metrics_list.append(metric(pred_output[:,:,-num_for_target:], x[:,:,-num_for_target:]))

        record(test_rec_mae, test_rec_rmse, test_rec_mape, rec_set[0], rec_set[1], rec_set[2], mode="PeMS")
        record(test_pred_mae, test_pred_rmse, test_pred_mape, pred_set[0], pred_set[1], pred_set[2], mode="PeMS")
        record(test_hispred_mae, test_hispred_rmse, test_hispred_mape, hispred_set[0], hispred_set[1],
               hispred_set[2], mode="PeMS")

        for i, key in enumerate(test_dict_loss.keys()):
            test_dict_loss[key].append(output_list[i])

    test_metrics_list = np.array(test_metrics_list)
    mtest_metrics_list = np.mean(test_metrics_list, axis=0)

    mtest_loss = []
    for i, key in enumerate(test_dict_loss.keys()):
        mtest_loss.append(np.mean(test_dict_loss[key]))

    log = 'Test Total Loss: {:.4f}\n' \
          'Test Pred Loss: {:.4f}\t\t\tTest Hispred Loss: {:.4f}\n' \
          'Test Recon Loss: {:.4f}\t\t\tTest Invariant Loss: {:.4f} \n' \
          'Test Nodefit Loss: {:.4f}\t\t\tTest Metric Loss: {:.4f} \n' \
          'Test L1 Loss: {:.4f}\t\t\tTest KL Loss: {:.4f} \n' \
          'Test Rec  Speed  MAE: {:.4f}\t\t\tRMSE: {:.4f}\t\t\tMAPE: {:.4f} \n' \
          'Test Hispred Speed  MAE: {:.4f}\t\t\tRMSE: {:.4f}\t\t\tMAPE: {:.4f}\n' \
          'Test Pred Speed  MAE: {:.4f}\t\t\tRMSE: {:.4f}\t\t\tMAPE: {:.4f}\n' \
          'Test Pred Total MAE: {:.4f}\t\t\tRMSE: {:.4f}\t\t\tMAPE: {:.4f}\n'
    logger.info(log.format(mtest_loss[0],
                           mtest_loss[1], mtest_loss[2],
                           mtest_loss[3], mtest_loss[4],
                           mtest_loss[5], mtest_loss[6],
                           mtest_loss[7], mtest_loss[8],

                           np.mean(test_rec_mae['speed']), np.mean(test_rec_rmse['speed']),
                           np.mean(test_rec_mape['speed']),

                           np.mean(test_hispred_mae['speed']), np.mean(test_hispred_rmse['speed']),
                           np.mean(test_hispred_mape['speed']),

                           np.mean(test_pred_mae['speed']), np.mean(test_pred_rmse['speed']),
                           np.mean(test_pred_mape['speed']),

                           mtest_metrics_list[0], mtest_metrics_list[2], mtest_metrics_list[1],
                           ))

    return mtest_loss[1], mtest_metrics_list[0], mtest_metrics_list[2], mtest_metrics_list[1]


def baseline_test(runid,
                  model,
                  dataloader,
                  device,
                  logger,
                  cfg,
                  writer=None):

    scalar = dataloader['scalar']

    engine = Trainer(model=model,
                     base_lr=cfg['train']['base_lr'],
                     weight_decay=cfg['train']['weight_decay'],
                     milestones=cfg['train']['milestones'],
                     lr_decay_ratio=cfg['train']['lr_decay_ratio'],
                     min_learning_rate=cfg['train']['min_learning_rate'],
                     max_grad_norm=cfg['train']['max_grad_norm'],
                     num_for_target=cfg['data']['num_for_target'],
                     num_for_predict=cfg['data']['num_for_predict'],
                     loss_weight=cfg['train']['loss_weight'],
                     scaler=scalar,
                     device=device,
                     rec=cfg['train']['rec'],
                     pred=cfg['train']['pred'],
                     hispred=cfg['train']['hispred'],
                     regularize_rank=cfg['train']['regularize_rank'],
                     orthogonal=cfg['train']['orthogonal'],
                     sparse=cfg['train']['sparse'],
                     eig_rank=cfg['train']['eig_rank'],
                     )

    total_param = 0
    logger.info('Net\'s state_dict:')
    for param_tensor in engine.model.state_dict():
        logger.info(param_tensor + '\t' + str(engine.model.state_dict()[param_tensor].size()))
        total_param += np.prod(engine.model.state_dict()[param_tensor].size())
    logger.info('Net\'s total params:{:d}\n'.format(int(total_param)))

    logger.info('Optimizer\'s state_dict:')
    for var_name in engine.optimizer.state_dict():
        logger.info(var_name + '\t' + str(engine.optimizer.state_dict()[var_name]))

    nParams = sum([p.nelement() for p in model.parameters()])
    logger.info('Number of model parameters is {:d}\n'.format(int(nParams)))

    best_mode_path = cfg['train']['best_mode']
    logger.info("loading {}".format(best_mode_path))

    save_dict = torch.load(best_mode_path)
    engine.model.load_state_dict(save_dict['model_state_dict'])
    logger.info('model load success! {}\n'.format(best_mode_path))


    loss_list = ['total', 'pred', 'hispred', 'recon', 'invariant', 'nodefit', 'metric', 'L1', 'KL']
    data_dim_list = ['speed']
    mvalid_pred_loss, mvalid_pred_mae, mvalid_pred_rmse, mvalid_pred_mape = model_val(runid,
                                                                                      engine=engine,
                                                                                      dataloader=dataloader,
                                                                                      device=device,
                                                                                      logger=logger,
                                                                                      cfg=cfg,
                                                                                      epoch=-1,
                                                                                      loss_list=loss_list,
                                                                                      data_dim_list=data_dim_list)

    mtest_total_loss, mtest_pred_mae, mtest_pred_rmse, mtest_pred_mape = model_test(runid, engine, dataloader, device,
                                                                                    logger, cfg,
                                                                                    loss_list=loss_list,
                                                                                    data_dim_list=data_dim_list,
                                                                                    mode='Test')

    return mvalid_pred_loss, mvalid_pred_mae, mvalid_pred_rmse, mvalid_pred_mape, \
           mtest_total_loss, mtest_pred_mae, mtest_pred_rmse, mtest_pred_mape

