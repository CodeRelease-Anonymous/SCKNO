#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import numpy as np
import torch
import copy
import sys
# sys.path.append('../..')
from torch.utils.data import Dataset, DataLoader



class StandardScaler_Torch:
    """
    Standard the input
    """

    def __init__(self, mean, std, device):
        self.mean = torch.tensor(data=mean, dtype=torch.float, device=device)
        self.std = torch.tensor(data=std, dtype=torch.float, device=device)

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean

def load_dataset(dataset_dir,
                 train_batch_size,
                 valid_batch_size=None,
                 test_batch_size=None,
                 logger=None,
                 device='cuda:0'):

    cat_data = np.load(dataset_dir)
    x_train = cat_data['x_train'].transpose((0, 2, 1, 3))
    y_train = cat_data['y_train'].transpose((0, 2, 1, 3))
    x_test = cat_data['x_test'].transpose((0, 2, 1, 3))
    y_test = cat_data['y_test'].transpose((0, 2, 1, 3))
    x_val = cat_data['x_val'].transpose((0, 2, 1, 3))
    y_val = cat_data['y_val'].transpose((0, 2, 1, 3))

    x_train = np.concatenate([x_train, y_train], axis=2)
    x_val = np.concatenate([x_val, y_val], axis=2)
    x_test = np.concatenate([x_test, y_test], axis=2)

    scaler = StandardScaler_Torch(x_train[...,:1].mean((0, 1, 2)),
                                  x_train[...,:1].std((0, 1, 2)),
                                  device=device)

    train_dataset = traffic_demand_prediction_dataset(x_train)

    val_dataset = traffic_demand_prediction_dataset(x_val)

    test_dataset = traffic_demand_prediction_dataset(x_test)

    dataloader = {}
    dataloader['train'] = DataLoader(dataset=train_dataset, shuffle=True, batch_size=train_batch_size)
    dataloader['val'] = DataLoader(dataset=val_dataset, shuffle=False, batch_size=valid_batch_size)
    dataloader['test'] = DataLoader(dataset=test_dataset, shuffle=False, batch_size=test_batch_size)

    dataloader['scalar'] = scaler

    if logger != None:
        logger.info(('train x', x_train.shape))

        logger.info('\n')
        logger.info(('val x', x_val.shape))

        logger.info('\n')
        logger.info(('test x', x_test.shape))

        logger.info('\n')
        logger.info('Speed scaler.mean : {}, scaler.std : {}'.format(scaler.mean,
                                                                     scaler.std))

    return dataloader


class traffic_demand_prediction_dataset(Dataset):
    def __init__(self, x):
        time = x[..., 1:]
        x = x[..., :1]
        self.x = torch.tensor(x).to(torch.float32)
        self.x_time = torch.tensor(time).to(torch.float32)

    def __getitem__(self, item):
         return self.x[item], self.x_time[item],

    def __len__(self):
        return self.x.shape[0]

def generate_time_one_hot(arr):
    dayofweek_len = 7
    timeofday_len = 288

    dayofweek = torch.eye(dayofweek_len, device=arr.device)[arr[..., 0].to(torch.int32)]
    timeofday = arr[..., 1].to(torch.int32)/timeofday_len
    arr = torch.cat([dayofweek, timeofday.unsqueeze(-1), arr[..., 2:]], dim=-1)
    return arr

def generate_time_one(arr):
    dayofweek_len = 7
    timeofday_len = 288

    dayofweek = torch.eye(dayofweek_len, device=arr.device)[arr[..., 0].to(torch.int32)]
    timeofday = torch.eye(timeofday_len, device=arr.device)[arr[..., 1].to(torch.int32)]
    arr = torch.cat([dayofweek, timeofday, arr[..., 2:]], dim=-1)
    return arr
