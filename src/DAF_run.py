# @Created   : 2022/10/15 21:41 PM
# @Author    : Sungmin Han
import os
import argparse
import numpy as np
from scipy import stats as stats

import torch
from torch.utils.data import DataLoader, Dataset

import utils.dataloader as loader
import DAF
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description="DAF")
    parser.add_argument('--batch_size', '-b', type=int, default=128)    
    parser.add_argument('--bootstrap_test', '-bt', type=int, default=100)
    parser.add_argument('--sampling_percent', '-sp', type=float, default=1.0)
    parser.add_argument('--number_iteration', '-ni', type=int, default=1600)    
    parser.add_argument('--corr_thres', '-ct', type=float, default=0.5)
    parser.add_argument('--pval_thres', '-pt', type=float, default=0.0001)
    parser.add_argument('--testingset', '-ts', type=str, default='ISCX2012')
    parser.add_argument('--flatten', '-fl', type=bool, default=True)
    parser.add_argument('--randomize', '-rz', type=bool, default=True)
    
    args = parser.parse_args()
    params = vars(args)
    
    batch_size = params['batch_size']
    flatteness = params['flatten']
    randomized = params['randomize']
    
    
    testing_dataset_idx_path = './'
    testing_dataset_path = './'
    testing_dataset_corr_map_path = './'

    D2_dataset_idx_path = './'
    D2_dataset_path = './'
    
    testing_set_idx = np.load(testing_dataset_idx_path)
    corr_map = np.load(testing_dataset_corr_map_path)
        
    test_dataset = loader.PklsFolder(testing_dataset_path,params['testingset'],flatten=flatteness,randomize = randomized)
    D2_dataset_idx = np.load(D2_dataset_idx_path)
    D2_dataset = loader.PklsFolder(D2_dataset_path,params['testingset'],flatten=flatteness,randomize = randomized)
    
    data_random_sampler = torch.utils.data.SubsetRandomSampler(testing_set_idx)
    test_dataloader = DataLoader(test_dataset,batch_size = batch_size,shuffle = False,sampler = data_random_sampler)
    
    deceptive_features = DAF(params['number_iteration'],params['pval_thres'],
                             params['bootstrap_test'],params['corr_thres'],corr_map,test_dataloader,
                             D2_dataset_idx,D2_dataset,params['sampling_percent'],batch_size)
    