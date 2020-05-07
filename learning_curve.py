# -*- coding: utf-8 -*-
"""
Created on Wed May  6 22:37:07 2020

@author: shiyl
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

rfd = 'D:/git/results/'


#%% Evolution

def myplot(file_name):
    
    [train_losses,val_losses,train_accs, val_accs] = np.load(file_name)
    
    fig, (ax1,ax2) = plt.subplots(1, 2, figsize=(20, 10))  
    ax1.plot(train_losses,marker=".")
    ax1.plot(val_losses,marker=".")
    ax1.grid()
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax1.set_xlabel('# Epoch')
    ax1.set_ylabel('Score')
    ax1.set_title('Loss')
    
    
    ax2.plot(train_accs,marker=".")
    ax2.plot(val_accs,marker=".")
    ax2.grid()
    ax2.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax2.set_xlabel('# Epoch')
    ax2.set_ylabel('Score')
    ax2.set_title('Accuracy')
    
    
    plt.savefig(file_name.rsplit('.',1)[0] +'.png')


#%%

# file_name = '../results/loss_train2_part1.npy'
# myplot(file_name)




