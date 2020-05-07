# -*- coding: utf-8 -*-
"""
Created on Wed May  6 22:37:07 2020

@author: shiyl
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

rfd = 'D:/git/results/'    
    # [train_losses,val_losses,train_accs, val_accs] = np.load(rfd + file_name)
    
    # fig, (plt,ax2) = plt.subplots(1, 2, figsize=(20, 10))  
    # plt.plot(train_losses,marker=".")
    # plt.plot(val_losses,marker=".")
    # plt.grid()
    # plt.xaxis.set_major_locator(MaxNLocator(integer=True))
    # plt.set_xlabel('# Epoch')
    # plt.set_ylabel('Score')
    # plt.set_title('Loss')
     

# learning_curve(rfd + 'loss_across_epochs.npy')


#%% Evaluation

file_names = sorted(os.listdir(rfd)) 

train_accs, test_accs = [], []

parts = [1,2,4,8,16][::-1] 

for partition in parts:
    
    train_fn = [f for f in file_names if 'part'+str(partition) in f and 'train' in f] 
    test_fn = [f for f in file_names if 'part'+str(partition) in f and 'test' in f] 
    
    train_acc = np.load(rfd + train_fn[0])[2,-1]
    test_acc =  np.load(rfd + test_fn[0])
    
    train_accs.append(train_acc)
    test_accs.append(test_acc)


n_train = 60000*.15
n_parts = [n_train/part for part in parts]


plt.figure(figsize=(20, 10))  
plt.loglog(n_parts,train_accs,marker=".")
plt.loglog(n_parts,test_accs,marker=".")
plt.grid()
# plt.xaxis.set_major_locator(MaxNLocator(integer=True))
plt.xticks(label=[str(part) for part in parts])
plt.xlabel('Training data size')
plt.ylabel('Acccuracy')
plt.legend({'Train','Test'})
# plt.set_title('Loss')


plt.savefig(rfd + 'data_partition.png')




