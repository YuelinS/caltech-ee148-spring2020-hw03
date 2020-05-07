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


#%% Incorrect examples

[imgs,trues,preds] = np.load(rfd + 'incorrect_examples.npy')

fig,axs = plt.subplots(3,3,figsize=(15,15))
axs = axs.ravel()
for i in range(9):  
    axs[i].imshow(np.squeeze(imgs[i]),cmap = 'gray')
    axs[i].set_title(f'True: {trues[i]}, Pred: {preds[i][0]}')
    
plt.savefig(rfd + 'incorrect_examples.png')


#%% kernels

# see main.py bottom


#%% confusion matrix
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

files = np.load(rfd + 'confusion_matrix_data.npz')

trues = files['arr_0'].ravel()
preds = np.squeeze(files['arr_1']).ravel()

mat = confusion_matrix(trues,preds)

plt.figure(figsize=(20,20))
disp = ConfusionMatrixDisplay(confusion_matrix=mat,
                                  display_labels=[str(i) for i in range(10)])
disp.plot(cmap = 'GnBu',values_format = '.0f')
plt.savefig(rfd + 'confusion_matrix.png')










