# @Created   : 2022/10/15 21:41 PM
# @Author    : Sungmin Han
import os
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

def corr_idx(correlation_map,feature_list,thres):
    temp = correlation_map
    corr_list = []
    for idx in feature_list:
        start_idx = np.where(temp[idx].argsort()[::-1]==idx)[0][0] + 1        
        corr_index_arr = temp[idx].argsort()[::-1][temp[idx][temp[idx].argsort()[::-1]] >= thres]
        corr_index_arr.sort()
        corr_list.append(corr_index_arr)
    return np.array(corr_list)

def erase_list_func(x,idx_list):
    meta_x = x.reshape(-1,1600)    
    mask_list = []
    for idx in idx_list:
        x = meta_x.clone()
        x[:,idx] = torch.randint(256,(len(idx),)).float()
        mask_list.append(x)
    return mask_list

def erase_list(x,idx):
    x = x.reshape(-1,1600)
    if len(idx) != 0:
        x[:,idx] = torch.randint(256,(len(idx),)).float()
    return x

def bootstrap_loader(d2_dataset,idx,percent,batchsize):
    
    if percent > 1 or percent <= 0:
        assert False, 'Check sampling percent'
        
    sampled_idx = np.random.choice(idx, int(np.trunc(idx.shape[0]*percent)),replace=True)
    random_sampler = torch.utils.data.SubsetRandomSampler(sampled_idx)
    
    return DataLoader(d2_dataset, batch_size = batchsize,sampler = random_sampler, shuffle = False)