# @Created   : 2022/10/15 21:41 PM
# @Author    : Sungmin Han
import os
import argparse
import numpy as np
from scipy import stats as stats

import torch
from torch.utils.data import DataLoader, Dataset

from sklearn import metrics
from sklearn.metrics import roc_curve
from sklearn import tree
from utils.daf_utils import corr_idx, erase_list_func, erase_list, bootstrap_loader

from threading import Thread
def check_bias(id,clf_bt,test_x,test_y,auc_list):
    auc = metrics.roc_auc_score(test_y, clf_bt.predict_proba(test_x)[:,1])
    auc_list.append([id,auc])
    return

def DAF(t_iter,p_thres,number_test,corr_thres,corr_map,test_loader,D2_dataset,D2_dataset_idx,sampling_percent,batch_size):
    
    X_train = torch.zeros(len(train_idx),1600)
    Y_train = torch.zeros(len(train_idx))
    count = 0
    for batch_id, (x, y) in enumerate(tqdm(test_loader)):
        b_size = x.shape[0]
        X_test[count:count+b_size,:] = x.reshape(x.shape[0],-1)
        Y_test[count:count+b_size] = y
        count += b_size
    count = 0

    deceptive_feature = []
    anchor_list = []
    mask_dt = []
    iteration = 0
    standard = 4
    no_detected_flag = False

    for iteration in range(1,t_iter):

        masked_x = daf_util.erase_list(X_train.clone(),deceptive_feature)    
        clf = tree.DecisionTreeClassifier(criterion='entropy',max_features=None,random_state=0)
        clf = clf.fit(masked_x, Y_train)
        ipt = (clf.feature_importances_)    
        ipt_index_list = ipt.argsort()[::-1][:standard]    

        testing_index_list = []
        for i in range(len(ipt_index_list)):
            if (np.in1d(ipt_index_list[i],deceptive_feature)) == False:
                testing_index_list.append(ipt_index_list[i])

        erase_candi = corr_idx(corr_map,testing_index_list,corr_thres)
        masked_X_train_list = erase_list_func(masked_x.clone(),erase_candi)

        if no_detected_flag == False:
            mask_dt = []
            for i in tqdm(range(len(masked_X_train_list))):
                clf_mask = tree.DecisionTreeClassifier(criterion='entropy',max_features=None,random_state=0)
                clf_mask = clf_mask.fit(masked_X_train_list[i],Y_train)
                mask_dt.append(clf_mask)
        else:
            clf_mask = tree.DecisionTreeClassifier(criterion='entropy',max_features=None,random_state=0)
            clf_mask = clf_mask.fit(masked_X_train_list[-1],Y_train)
            mask_dt.append(clf_mask)

        auc_array = np.zeros((standard+1,number_test))
        for sample_num in tqdm(range(number_test)):
            
            boot_loader = bootstrap_loader(D2_dataset,D2_dataset_idx,sampling_percent,batch_size)
            
            x_list,y_list = list(),list()            
            for x, y in boot_loader: 
                x_list.extend(x.reshape(x.shape[0],-1).tolist())
                y_list.extend(y.tolist())

            X_val_D2 = torch.as_tensor(x_list)
            Y_val_D2 = torch.as_tensor(y_list)

            del boot_loader

            auc = metrics.roc_auc_score(Y_val_D2, clf.predict_proba(X_val_D2)[:,1])
            auc_array[0,sample_num] = auc

            masked_auc = []
            thres = []
            for i in range(len(mask_dt)):
                masked_X_val_D2 = erase_list(X_val_D2.clone(),erase_candi[i])            
                thres.append(Thread(target=check_bias, args=(i+1,mask_dt[i],masked_X_val_D2,Y_val_D2,masked_auc)))
                thres[i].start()
            for i in range(len(mask_dt)):
                thres[i].join()

            masked_auc = np.array(masked_auc)   
            masked_auc = masked_auc[masked_auc[:,0].argsort()]
            masked_auc = masked_auc[:,1]

            for i in range(len(mask_dt)):
                auc_array[i+1,sample_num] = masked_auc[i]

        p_val_list = []
        for i in range(1,standard+1):
            _,pval = stats.ttest_rel(auc_array[0,:],auc_array[i,:],alternative='less')
            p_val_list.append(pval)

        p_val_list = np.array(p_val_list)

        min_pval = p_val_list.min()
        anchor_idx = p_val_list.argmin()

        auc_mean_array = auc_array.mean(axis=1)
        org_auc_mean = auc_mean_array[0]
        mask_auc_mean = auc_mean_array[anchor_idx+1]

        if np.isnan(min_pval):
            print('P-val is Nan')
            anchor_idx = auc_mean_array[1:].argmin()
            mask_auc_mean = auc_mean_array[anchor_idx+1]
            if org_auc_mean > mask_auc_mean:
                print('No detected feature')
                standard += 1
                no_detected_flag = True        
                continue        
        elif min_pval >= p_thres:
            print('No detected feature')
            standard += 1
            no_detected_flag = True        
            continue

        erase_candi = corr_idx(corr_map,[testing_index_list[anchor_idx]],corr_thres)[0]
        anchor_list.append(testing_index_list[anchor_idx])
        deceptive_feature.extend(erase_candi)
        deceptive_feature = list(set(deceptive_feature))
        no_detected_flag = False
        
    return deceptive_feature