# -*- coding: utf-8 -*-
"""
Created on Sat Jun 10 20:35:21 2023

@author: mkim332
"""

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment as linear_assignment  
from scipy.stats import pearsonr

def clustering_acc(Y_pred, Y, 
                   title_ = "",
                   logger = None):
    # Based on Hungarian algorithm (Assignment problem)
    '''
    This is implemented to estimate accuracy of clustering in last sequence only
    - Assume the labels of prediction and ground-truth are not matching - Mixed
    
    Shape of inputs (e.g., argmaxed softmax logits and integer type ground truth labels): (sample size, )
    '''
    Y_pred = Y_pred.astype(int)
    Y = Y.astype(int)
    Y_pred = Y_pred.reshape(-1)
    Y = Y.reshape(-1)

    # Get a total number of clusters in inputs
    D = max(Y_pred.max(), Y.max())+1
    # form a matrix of 6 x 6
    w = np.zeros((D,D), dtype=np.int64)
    # matching labels - This is done by scoring of the repeats in sequential labels
    for i in range(Y_pred.size):
        w[Y_pred[i], Y[i]] += 1
    ind = linear_assignment(w.max() - w)
    acc_ = (sum([w[ind[0][i],ind[1][i]] for i in range (D)])*1.0/Y_pred.size) * 100
    if logger is None:
        print_ = print
    else:
        print_ = logger.info
        
    print_(f"Clustering ACC {title_}: {np.round(acc_,5)}%")
    
def clustering_purity(predicted_labels, true_labels):
    # Count the number of samples in each cluster
    cluster_counts = {}
    for predlabel, true_label in zip(predicted_labels, true_labels):
        if predlabel not in cluster_counts:
            cluster_counts[predlabel] = {}
        if true_label not in cluster_counts[predlabel]:
            cluster_counts[predlabel][true_label] = 0
        cluster_counts[predlabel][true_label] += 1
    
    # Calculate purity
    total_samples = len(predicted_labels)
    purity = 0.0
    for cluster, counts in cluster_counts.items():
        max_count = max(counts.values())
        purity += max_count
    
    purity /= total_samples
    return purity
def MA(x, y, window_ = 3):
    length_ = x.shape[0]
    x_ = np.array(x)
    y_ = np.array(y)
    
    x = np.pad(x, (window_//2, window_//2), mode='constant')
    y = np.pad(y, (window_//2, window_//2), mode='constant')
    for i in range (window_//2, length_ + (window_//2)):
        x_[i-window_//2] = x[i-(window_//2):i+(window_//2)].mean()
        y_[i-window_//2] = y[i-(window_//2):i+(window_//2)].mean()
    return x_, y_

def forecasting_acc(x_fore, x_target, 
                    x_init, target_init,
                    sta_reg = 25, avg = 3):
    '''
    Compute quantitative measures in full sequence-wise 
    '''
    ''' 
    unnormalized inputs 
    x_fore and x_true -->(length, feature)   the sequence includes non-forecasted region for which acc is not computed.
    need to accumulate at where it gets called and take a mean of that
    '''
    # # Sample-wise normalization
    # x_fore = (x_fore - x_fore.mean(0)) / x_fore.std(0)
    # x_target = (x_target - x_target.mean(0)) / x_target.std(0)
    # # Calibration with respect to the sequence up to stationarization of target (excluding forecasted regions)
    # ref = x_target[:sta_reg,:].mean(0)
    # calib = ref - x_fore[:sta_reg,:].mean(0)
    # x_fore = x_fore + calib[None,:] # shift by calib 
    
    # # Compute acc over forecasted ones
    # x_fore = x_fore[sta_reg:,:]
    # x_target = x_target[sta_reg:,:]
    
    
    # Sample-wise normalization
    x_fore = (x_fore - x_init.mean(0)) / x_init.std(0) # standardization
    # cali_ = x_fore[:40,:].mean(0) # calibration based along the healthy state 
    # x_fore -= cali_
    x_fore = x_fore[:,:]
    
    x_target = (x_target - target_init.mean(0)) / target_init.std(0)
    # cali_ = x_target[:40,:].mean(0)
    # x_target -= cali_
    x_target = x_target[:,:]
    
    ## MSE
    mse_ = np.mean((x_fore - x_target) ** 2)
    ## MAE
    mae_ = np.mean(np.abs(x_fore - x_target))
    ## MAPE
    mape_ = np.mean(np.abs((x_fore - x_target) / x_target)) #* 100
    # PCC
    for ii in range (avg):
        # Moving average for computing PCC over trend
        x_fore, x_target = MA(x_fore, x_target)
    correlations = [pearsonr(x_fore[:, i], x_target[:, i])[0] for i in range(x_fore.shape[1])]
    pcc = np.mean(correlations)

    return mse_, mae_, mape_, pcc
# # Example usage
# predicted_clusters = np.array([1, 1, 2, 2, 2, 3])
# true_labels = np.array([1, 1, 2, 2, 3, 3])

# nmi = normalized_mutual_info_score(true_labels, predicted_clusters)
# ari = adjusted_rand_score(true_labels, predicted_clusters)
# purity = clustering_purity(predicted_clusters, true_labels)

# print("Normalized Mutual Information (NMI):", nmi)
# print("Adjusted Rand Index (ARI):", ari)
# print("Clustering Purity (ACC):", purity)



def RUL_metric(pred, target): # TODO 
    # pred and traget dim --> (samples_num, 1) estimation at the last time step of incomplete sequences
    
    assert pred.shape == target.shape
    instances_num = pred.shape[0]
    # If normalization was applied put them back
    pred = pred * 125.
    target = target * 125.
    
    error_ = pred - target
    error_ = error_.view(-1)
        
    # RMSE 
    rmse_ = torch.round((((error_**2).sum()) / instances_num)**0.5, decimals = 5)

    # SCORE
    s = torch.zeros(instances_num)
    for i in range (instances_num):
        if error_[i] < 0:
            s[i] = torch.exp(-error_[i]/13) - 1.
        elif error_[i] >= 0:
            s[i] = torch.exp(error_[i]/10) - 1.
    score_ = torch.round(s.sum(), decimals = 5)
    return rmse_.detach().cpu().numpy(), score_.detach().cpu().numpy(), error_.detach().cpu().numpy()