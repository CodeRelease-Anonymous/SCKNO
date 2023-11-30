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
    cat_data = np.load(dataset_dir, allow_pickle=True)
    all_data = {
        'test': {
            # [batch, node_num, time, dim]
            'x': cat_data['test_x'],
            'x_time': (cat_data['test_x_time']),
        },
        'scaler':{
            'mean': cat_data['scaler_mean'],
            'std': cat_data['scaler_std'],
        },
    }

    scaler = StandardScaler_Torch(all_data['scaler']['mean'],
                                  all_data['scaler']['std'],
                                  device=device)

    test_dataset = traffic_demand_prediction_dataset(all_data['test']['x'],
                                                     all_data['test']['x_time'],
                                                     )

    dataloader = {}
    dataloader['test'] = DataLoader(dataset=test_dataset, shuffle=False, batch_size=test_batch_size)
    dataloader['scalar'] = scaler

    if logger != None:
        logger.info('\n')
        logger.info(('test x', all_data['test']['x'].shape))
        logger.info(('test x time', all_data['test']['x_time'].shape))

        logger.info('\n')
        logger.info('Speed scaler.mean : {}, scaler.std : {}'.format(scaler.mean,
                                                                     scaler.std))
    return dataloader


class traffic_demand_prediction_dataset(Dataset):
    def __init__(self, x, x_time):
        time = x_time[..., :2]
        weather = x_time[..., 2:]
        # time = self.__generate_one_hot(time)
        x_time = np.concatenate([time, weather], axis=-1)

        self.x = torch.tensor(x).to(torch.float32)
        self.x_time = torch.tensor(x_time).to(torch.float32)
        # self.x_time = torch.repeat_interleave(self.x_time.unsqueeze(dim=1), repeats=self.x.shape[1], dim=1)

    def __getitem__(self, item):
         return self.x[item], self.x_time[item],

    def __len__(self):
        return self.x.shape[0]

def generate_time_one_hot(arr):
    dayofweek_len = 7
    timeofday_len = 48

    dayofweek = torch.eye(dayofweek_len, device=arr.device)[arr[..., 0].to(torch.int32)]
    timeofday = torch.eye(timeofday_len, device=arr.device)[arr[..., 1].to(torch.int32)]
    arr = torch.cat([dayofweek, timeofday, arr[..., 2:]], dim=-1)
    return arr
