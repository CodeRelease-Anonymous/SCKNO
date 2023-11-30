#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import math

import numpy as np
import torch
import scipy.stats as stats


def masked_rmse_np(preds, labels, null_val=np.nan):
    return np.sqrt(masked_mse_np(preds=preds, labels=labels, null_val=null_val))


def masked_mse_np(preds, labels, null_val=np.nan):
    with np.errstate(divide='ignore', invalid='ignore'):
        if np.isnan(null_val):
            mask = ~np.isnan(labels)
        else:
            mask = np.not_equal(labels, null_val)
        mask = mask.astype('float32')
        mask /= np.mean(mask)
        rmse = np.square(np.subtract(preds, labels)).astype('float32')
        rmse = np.nan_to_num(rmse * mask)
        return np.mean(rmse)


def masked_mae_np(preds, labels, null_val=np.nan, mode='dcrnn'):
    with np.errstate(divide='ignore', invalid='ignore'):
        if np.isnan(null_val):
            mask = ~np.isnan(labels)
        else:
            mask = np.not_equal(labels, null_val)
        mask = mask.astype('float32')
        mask /= np.mean(mask)
        mae = np.abs(np.subtract(preds, labels)).astype('float32')
        mae = np.nan_to_num(mae * mask)
        if mode == 'dcrnn':
            return np.mean(mae)
        else:
            return np.mean(mae, axis=(0, 1))


def masked_mape_np(preds, labels, null_val=np.nan):
    with np.errstate(divide='ignore', invalid='ignore'):
        if np.isnan(null_val):
            mask = ~np.isnan(labels)
        else:
            mask = np.not_equal(labels, null_val)
        mask = mask.astype('float32')
        mask /= np.mean(mask)

        preds = preds[labels > 10]
        mask = mask[labels > 10]
        labels = labels[labels > 10]

        mape = np.abs(np.divide(np.subtract(preds, labels).astype('float32'), labels))
        mape = np.nan_to_num(mask * mape)
        return np.mean(mape) * 100


def masked_mse_torch(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels != null_val)
    mask = mask.float()
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = (preds - labels) ** 2
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


def masked_rmse_torch(preds, labels, null_val=np.nan):
    return torch.sqrt(masked_mse_torch(preds=preds, labels=labels, null_val=null_val))


def masked_mae_torch(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels != null_val)
    mask = mask.float()
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds - labels)
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


def masked_mape_torch(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels != null_val)
    mask = mask.float()
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)

    preds = preds[labels > 10]
    mask = mask[labels > 10]
    labels = labels[labels > 10]

    loss = torch.abs(preds - labels) / labels
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss) * 100


def metric(pred, real):
    mae = masked_mae_torch(pred, real, np.inf).item()
    mape = masked_mape_torch(pred, real, np.inf).item()
    rmse = masked_rmse_torch(pred, real, np.inf).item()
    return mae, mape, rmse


def metric_np(pred, real):
    mae = masked_mae_np(pred, real, np.inf)
    mape = masked_mape_np(pred, real, np.inf)
    rmse = masked_rmse_np(pred, real, np.inf)
    return mae, mape, rmse


def metric_all(preds, reals, mode='BJ'):
    time = preds[0].shape[2]

    mae = {}
    rmse = {}
    mape = {}

    if mode == "BJ":
        mae['bike'] = np.zeros(time)
        mae['taxi'] = np.zeros(time)
        mae['bus'] = np.zeros(time)
        mae['speed'] = np.zeros(time)

        rmse['bike'] = np.zeros(time)
        rmse['taxi'] = np.zeros(time)
        rmse['bus'] = np.zeros(time)
        rmse['speed'] = np.zeros(time)

        mape['bike'] = np.zeros(time)
        mape['taxi'] = np.zeros(time)
        mape['bus'] = np.zeros(time)
        mape['speed'] = np.zeros(time)

    elif mode == 'NYC':
        mae['bike'] = np.zeros(time)
        mae['taxi'] = np.zeros(time)

        rmse['bike'] = np.zeros(time)
        rmse['taxi'] = np.zeros(time)

        mape['bike'] = np.zeros(time)
        mape['taxi'] = np.zeros(time)
    elif mode == 'PeMS':
        mae['speed'] = np.zeros(time)
        rmse['speed'] = np.zeros(time)
        mape['speed'] = np.zeros(time)

    if len(preds) > 1:
        for t in range(time):
            if mode=='BJ':
                mae['bike'][t], mape['bike'][t], rmse['bike'][t] = metric(preds[0][:, :, t, :], reals[0][:, :, t, :])
                mae['taxi'][t], mape['taxi'][t], rmse['taxi'][t] = metric(preds[1][:, :, t, :], reals[1][:, :, t, :])
                mae['bus'][t], mape['bus'][t], rmse['bus'][t] = metric(preds[2][:, :, t, :], reals[2][:, :, t, :])
                mae['speed'][t], mape['speed'][t], rmse['speed'][t] = metric(preds[3][:, :, t, :], reals[3][:, :, t, :])
            elif mode=='NYC':
                mae['bike'][t], mape['bike'][t], rmse['bike'][t] = metric(preds[0][:, :, t, :], reals[0][:, :, t, :])
                mae['taxi'][t], mape['taxi'][t], rmse['taxi'][t] = metric(preds[1][:, :, t, :], reals[1][:, :, t, :])
            elif mode=='PeMS':
                mae['speed'][t], mape['speed'][t], rmse['speed'][t] = metric(preds[0][:, :, t, :], reals[0][:, :, t, :])
    else:
        for t in range(time):
            mae['speed'][t], mape['speed'][t], rmse['speed'][t] = metric(preds[0][:, :, t, :], reals[0][:, :, t, :])

    return mae, rmse, mape


def record(all_mae, all_rmse, all_mape, mae, rmse, mape, only_last=False, mode="BJ"):
    if only_last:

        if mode=='BJ':
            all_mae['bike'].append(mae['bike'][-1])
            all_mae['taxi'].append(mae['taxi'][-1])
            all_mae['bus'].append(mae['bus'][-1])
            all_mae['speed'].append(mae['speed'][-1])

            all_rmse['bike'].append(rmse['bike'][-1])
            all_rmse['taxi'].append(rmse['taxi'][-1])
            all_rmse['bus'].append(rmse['bus'][-1])
            all_rmse['speed'].append(rmse['speed'][-1])

            all_mape['bike'].append(mape['bike'][-1])
            all_mape['taxi'].append(mape['taxi'][-1])
            all_mape['bus'].append(mape['bus'][-1])
            all_mape['speed'].append(mape['speed'][-1])

        if mode=="NYC":
            all_mae['bike'].append(mae['bike'][-1])
            all_mae['taxi'].append(mae['taxi'][-1])

            all_rmse['bus'].append(rmse['bus'][-1])
            all_rmse['speed'].append(rmse['speed'][-1])

            all_mape['bus'].append(mape['bus'][-1])
            all_mape['speed'].append(mape['speed'][-1])

        elif mode=="PeMS":

            all_mae['speed'].append(mae['speed'][-1])
            all_rmse['speed'].append(rmse['speed'][-1])
            all_mape['speed'].append(mape['speed'][-1])

    else:

        if mode == 'BJ':
            all_mae['bike'].append(np.mean(mae['bike']))
            all_mae['taxi'].append(np.mean(mae['taxi']))
            all_mae['bus'].append(np.mean(mae['bus']))
            all_mae['speed'].append(np.mean(mae['speed']))

            all_rmse['bike'].append(np.mean(rmse['bike']))
            all_rmse['taxi'].append(np.mean(rmse['taxi']))
            all_rmse['bus'].append(np.mean(rmse['bus']))
            all_rmse['speed'].append(np.mean(rmse['speed']))

            all_mape['bike'].append(np.mean(mape['bike']))
            all_mape['taxi'].append(np.mean(mape['taxi']))
            all_mape['bus'].append(np.mean(mape['bus']))
            all_mape['speed'].append(np.mean(mape['speed']))

        if mode == "NYC":
            all_mae['bike'].append(np.mean(mae['bike']))
            all_mae['taxi'].append(np.mean(mae['taxi']))

            all_rmse['bike'].append(np.mean(rmse['bike']))
            all_rmse['taxi'].append(np.mean(rmse['taxi']))

            all_mape['bike'].append(np.mean(mape['bike']))
            all_mape['taxi'].append(np.mean(mape['taxi']))

        elif mode == "PeMS":

            all_mae['speed'].append(np.mean(mae['speed']))
            all_rmse['speed'].append(np.mean(rmse['speed']))
            all_mape['speed'].append(np.mean(mape['speed']))
