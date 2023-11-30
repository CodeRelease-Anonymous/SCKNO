#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import copy
from datetime import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import sys
import os
from tqdm import tqdm

from ModelTest_BJ import model_val, model_test
from helper_BJ import Trainer
from tools.metrics import record, metric

sys.path.append(os.getcwd())
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..")))


def baseline_train(runid,
                   model,
                   dataloader,
                   device,
                   logger,
                   cfg,
                   writer=None,
                   ):
    print("start training...", flush=True)
    save_path = os.path.join('Results', cfg['data']['name'], cfg['model_name'], 'exp{:s}'.format(cfg['expid']), 'ckpt')
    scalar = dataloader['scalar']

    num_for_predict = cfg['data']['num_for_predict']
    num_for_target = cfg['data']['num_for_target']
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

    if cfg['train']['load_initial']:
        best_mode_path = cfg['train']['best_mode']
        logger.info("loading {}".format(best_mode_path))
        save_dict = torch.load(best_mode_path)['model_state_dict']

        for param in engine.model.state_dict():
            try:
                if save_dict[param].shape != engine.model.state_dict()[param].shape:
                    save_dict.pop(param)
                    logger.info('remove: {:s}'.format(param))
            except:
                logger.info("origin model is missing the parameter: {:s}".format(param))
                continue

        engine.model.load_state_dict(save_dict, strict=False)
        logger.info('model load success! {}'.format(best_mode_path))

    else:
        logger.info('Start training from scratch!')
        save_dict = dict()

    begin_epoch = cfg['train']['epoch_start']
    epochs = cfg['train']['epochs']
    tolerance = cfg['train']['tolerance']

    his_loss = []
    val_time = []
    train_time = []
    best_val_loss = float('inf')
    best_epoch = -1
    stable_count = 0
    kl_weight_upper = 1.0

    # Define parameters for annealing kl_weight
    anneal_time = 5
    T = anneal_time*len(dataloader['train'])
    anneal_count = 0

    logger.info('begin_epoch: {}, total_epochs: {}, patient: {}, best_val_loss: {:.4f}'.
                format(begin_epoch, epochs, tolerance, best_val_loss))

    loss_list = ['total', 'pred', 'hispred', 'recon', 'invariant', 'nodefit', 'metric', 'L1', 'KL']
    data_dim_list = ['bike', 'taxi', 'bus', 'speed']
    for epoch in range(begin_epoch, begin_epoch + epochs + 1):

        train_loss_name = 'train_{:s}_loss'

        train_dict_loss = {}
        for name in loss_list:
            train_dict_loss[train_loss_name.format(name)] = []

        train_rec_mape = {}
        train_rec_rmse = {}
        train_rec_mae = {}

        train_pred_mape = {}
        train_pred_rmse = {}
        train_pred_mae = {}

        train_hispred_mape = {}
        train_hispred_rmse = {}
        train_hispred_mae = {}

        train_total_list = []

        for name in data_dim_list:
            train_rec_mae[name] = []
            train_rec_rmse[name] = []
            train_rec_mape[name] = []
            train_pred_mae[name] = []
            train_pred_rmse[name] = []
            train_pred_mape[name] = []
            train_hispred_mae[name] = []
            train_hispred_rmse[name] = []
            train_hispred_mape[name] = []

        t1 = time.time()

        train_dataloder = dataloader['train']
        train_tqdm_loader = tqdm(enumerate(train_dataloder))

        if epoch < 100:
            kl_weight = 1e-6
        else:
            anneal_count += 1
            kl_weight = min(kl_weight_upper, 1e-6 + kl_weight_upper * anneal_count / epochs)

        for iter, (x, pos) in train_tqdm_loader:

            x = x.to(device)

            tpos = pos[..., :3].to(device)
            weather = pos[..., 3:].to(device)

            output_list = engine.train(x, tpos, weather, kl_weight)

            rec_set = output_list[len(loss_list):len(loss_list)+3]
            hispred_set = output_list[len(loss_list)+3:len(loss_list)+6]
            pred_set = output_list[len(loss_list)+6:len(loss_list)+9]
            rec_output, hispred_output, pred_output = output_list[-3:]

            for i, key in enumerate(train_dict_loss.keys()):
                train_dict_loss[key].append(output_list[i])

            record(train_rec_mae, train_rec_rmse, train_rec_mape, rec_set[0], rec_set[1], rec_set[2])
            record(train_pred_mae, train_pred_rmse, train_pred_mape, pred_set[0], pred_set[1], pred_set[2])
            record(train_hispred_mae, train_hispred_rmse, train_hispred_mape, hispred_set[0], hispred_set[1], hispred_set[2])
            train_total_list.append(metric(pred_output[:,:,-num_for_target:],
                                           x[:,:,-num_for_target:]))

            # For the issue that the CPU memory increases while training. DO NOT know why, but it works.
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        engine.scheduler.step()

        t2 = time.time()
        train_time.append(t2 - t1)

        s1 = time.time()
        mvalid_pred_loss, mvalid_pred_mae, mvalid_pred_rmse, mvalid_pred_mape = model_val(runid,
                                                                                          engine=engine,
                                                                                          dataloader=dataloader,
                                                                                          device=device,
                                                                                          logger=logger,
                                                                                          cfg=cfg,
                                                                                          epoch=epoch,
                                                                                          loss_list=loss_list,
                                                                                          data_dim_list=data_dim_list
                                                                                          )
        s2 = time.time()
        val_time.append(s2 - s1)

        mtest_pred_loss, mtest_pred_mae, mtest_pred_rmse, mtest_pred_mape = model_test(runid,
                                                                                       engine=engine,
                                                                                       dataloader=dataloader,
                                                                                       device=device,
                                                                                       cfg=cfg,
                                                                                       logger=logger,
                                                                                       loss_list=loss_list,
                                                                                       data_dim_list=data_dim_list,
                                                                                       mode='Train')

        mtrain_loss = []
        for i, key in enumerate(train_dict_loss.keys()):
            mtrain_loss.append(np.mean(train_dict_loss[key]))

        train_total_list = np.array(train_total_list)
        mtrain_total_list = np.mean(train_total_list, axis=0)

        if (epoch - 1) % cfg['train']['print_every'] == 0:
            log = 'Epoch: {:03d}, Inference Time: {:.4f} secs'
            logger.info(log.format(epoch, (s2 - s1)))
            log = 'Epoch: {:03d}, Train Total Loss: {:.4f} Learning rate: {} KL weight: {}\n' \
                  'Train Pred Loss: {:.4f}\t\t\tTrain Hispred Loss: {:.4f}\n' \
                  'Train Recon Loss: {:.4f}\t\t\tTrain Invariant Loss: {:.4f} \n' \
                  'Train Nodefit Loss: {:.4f}\t\t\tTrain Metric Loss: {:.4f} \n' \
                  'Train L1 Loss: {:.4f}\t\t\tTrain KL Loss: {:.4f} \n' \
                  'Train Rec  Bike  MAE: {:.4f}\t\t\tRMSE: {:.4f}\t\t\tMAPE: {:.4f} \n' \
                  'Train Rec  Taxi  MAE: {:.4f}\t\t\tRMSE: {:.4f}\t\t\tMAPE: {:.4f} \n' \
                  'Train Rec  Bus   MAE: {:.4f}\t\t\tRMSE: {:.4f}\t\t\tMAPE: {:.4f} \n' \
                  'Train Rec  Speed MAE: {:.4f}\t\t\tRMSE: {:.4f}\t\t\tMAPE: {:.4f} \n\n' \
                  'Train Hispred Bike  MAE: {:.4f}\t\t\tRMSE: {:.4f}\t\t\tMAPE: {:.4f}\n' \
                  'Train Hispred Taxi  MAE: {:.4f}\t\t\tRMSE: {:.4f}\t\t\tMAPE: {:.4f}\n' \
                  'Train Hispred Bus   MAE: {:.4f}\t\t\tRMSE: {:.4f}\t\t\tMAPE: {:.4f}\n' \
                  'Train Hispred Speed MAE: {:.4f}\t\t\tRMSE: {:.4f}\t\t\tMAPE: {:.4f}\n' \
                  'Train Pred Bike  MAE: {:.4f}\t\t\tRMSE: {:.4f}\t\t\tMAPE: {:.4f}\n' \
                  'Train Pred Taxi  MAE: {:.4f}\t\t\tRMSE: {:.4f}\t\t\tMAPE: {:.4f}\n' \
                  'Train Pred Bus   MAE: {:.4f}\t\t\tRMSE: {:.4f}\t\t\tMAPE: {:.4f}\n' \
                  'Train Pred Speed MAE: {:.4f}\t\t\tRMSE: {:.4f}\t\t\tMAPE: {:.4f}\n' \
                  'Train Pred Total MAE: {:.4f}\t\t\tRMSE: {:.4f}\t\t\tMAPE: {:.4f}\n'
            logger.info(log.format(epoch, mtrain_loss[0], str(engine.scheduler.get_lr()), str(kl_weight),
                                   mtrain_loss[1], mtrain_loss[2],
                                   mtrain_loss[3], mtrain_loss[4],
                                   mtrain_loss[5], mtrain_loss[6],
                                   mtrain_loss[7], mtrain_loss[8],

                                   np.mean(train_rec_mae['bike']), np.mean(train_rec_rmse['bike']), np.mean(train_rec_mape['bike']),
                                   np.mean(train_rec_mae['taxi']), np.mean(train_rec_rmse['taxi']), np.mean(train_rec_mape['taxi']),
                                   np.mean(train_rec_mae['bus']), np.mean(train_rec_rmse['bus']), np.mean(train_rec_mape['bus']),
                                   np.mean(train_rec_mae['speed']), np.mean(train_rec_rmse['speed']), np.mean(train_rec_mape['speed']),

                                   np.mean(train_hispred_mae['bike']), np.mean(train_hispred_rmse['bike']), np.mean(train_hispred_mape['bike']),
                                   np.mean(train_hispred_mae['taxi']), np.mean(train_hispred_rmse['taxi']), np.mean(train_hispred_mape['taxi']),
                                   np.mean(train_hispred_mae['bus']), np.mean(train_hispred_rmse['bus']), np.mean(train_hispred_mape['bus']),
                                   np.mean(train_hispred_mae['speed']), np.mean(train_hispred_rmse['speed']), np.mean(train_hispred_mape['speed']),

                                   np.mean(train_pred_mae['bike']), np.mean(train_pred_rmse['bike']),
                                   np.mean(train_pred_mape['bike']),
                                   np.mean(train_pred_mae['taxi']), np.mean(train_pred_rmse['taxi']),
                                   np.mean(train_pred_mape['taxi']),
                                   np.mean(train_pred_mae['bus']), np.mean(train_pred_rmse['bus']),
                                   np.mean(train_pred_mape['bus']),
                                   np.mean(train_pred_mae['speed']), np.mean(train_pred_rmse['speed']),
                                   np.mean(train_pred_mape['speed']),

                                   mtrain_total_list[0], mtrain_total_list[2], mtrain_total_list[1],
                                   ))
        his_loss.append(mvalid_pred_loss)
        if mvalid_pred_loss < best_val_loss:

            best_val_loss = mvalid_pred_loss
            epoch_best = epoch
            stable_count = 0

            save_dict.update(model_state_dict=copy.deepcopy(engine.model.state_dict()),
                             epoch=epoch_best,
                             best_val_loss=best_val_loss)

            ckpt_name = "exp{:s}_epoch{:d}_val_loss:{:.2f}_mae:{:.2f}_rmse:{:.2f}_mape:{:.2f}.pth". \
                format(cfg['expid'], epoch, mtest_pred_loss, mtest_pred_mae, mtest_pred_rmse, mtest_pred_mape)
            best_mode_path = os.path.join(save_path, ckpt_name)
            torch.save(save_dict, best_mode_path)
            logger.info(f'Better model at epoch {epoch_best} recorded.')
            logger.info('Best model is : {}'.format(best_mode_path))
            logger.info('\n')

        else:
            stable_count += 1
            if stable_count > tolerance:
                break

    logger.info("Average Training Time: {:.4f} secs/epoch".format(np.mean(train_time)))
    logger.info("Average Inference Time: {:.4f} secs".format(np.mean(val_time)))

    bestid = np.argmin(his_loss)

    logger.info("Training finished")
    logger.info("The valid loss on best model is {:.4f}, epoch:{:d}".format(round(his_loss[bestid], 4), epoch_best))

    logger.info('Start the model test phase........')
    logger.info("loading the best model for this training phase {}".format(best_mode_path))
    save_dict = torch.load(best_mode_path)
    engine.model.load_state_dict(save_dict['model_state_dict'])

    mvalid_pred_loss, mvalid_pred_mae, mvalid_pred_rmse, mvalid_pred_mape = model_val(runid,
                                                                                      engine=engine,
                                                                                      dataloader=dataloader,
                                                                                      device=device,
                                                                                      logger=logger,
                                                                                      cfg=cfg,
                                                                                      epoch=epoch_best,
                                                                                      loss_list=loss_list,
                                                                                      data_dim_list=data_dim_list,
                                                                                      )

    mtest_pred_loss, mtest_pred_mae, mtest_pred_rmse, mtest_pred_mape = model_test(runid,
                                                                                   engine=engine,
                                                                                   dataloader=dataloader,
                                                                                   device=device,
                                                                                   cfg=cfg,
                                                                                   logger=logger,
                                                                                   mode='Test',
                                                                                   loss_list=loss_list,
                                                                                   data_dim_list=data_dim_list,
                                                                                   )

    return mvalid_pred_loss, mvalid_pred_mae, mvalid_pred_rmse, mvalid_pred_mape, \
           mtest_pred_loss, mtest_pred_mae, mtest_pred_rmse, mtest_pred_mape