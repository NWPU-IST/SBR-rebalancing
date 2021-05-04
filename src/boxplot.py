# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 14:16:48 2020

@author: xingyu
"""

import matplotlib.pyplot as plt

# x = ['Derby','Camel','Wicket','Ambari']
# y = [0.29,0.25,0.23, 0.21]
# z=[0.21,0.18, 0.17,0.16]
# w=[0.43,0.44,0.43,0.30]
# plt.plot(x,y,'.-')
# plt.plot(x,z,'.-')
# # plt.plot(x, w, '.-')
# plt.legend(['F1.','Prec.'])
# plt.xlabel('dataset')
# plt.ylabel('performance')
# plt.savefig(r'd:\xingyu\Desktop\performance.eps', format='eps',dpi=1000)

import pandas as pd
import os
import numpy as np



# plt.figure(figsize=(10, 4))
os.chdir(r'd:\xingyu\Desktop')

# df = pd.read_excel('工作簿2.xlsx', sheet_name='Camel')
# CNN = df.CNN.values
# Blsmt = df.Blsmt.values
# Adasyn = df.Adasyn.values
# NM = df.NM.values
# Mahakil = df.Mahakil.values
# Rose = df.Rose.values
# data = [CNN, Blsmt, Adasyn, NM, Mahakil, Rose]

# plt.subplot(121)
# plt.boxplot(data)
# plt.xticks(np.arange(1, 7),['CNN', 'Blsmt', 'Adasyn', 'NM', 'Mahakil', 'Rose'])
# plt.xlabel('(a) Camel')
# plt.ylabel('F1.')

# df = pd.read_excel('工作簿2.xlsx', sheet_name='Chromium')
# CNN = df.CNN.values
# Blsmt = df.Blsmt.values
# Adasyn = df.Adasyn.values
# NM = df.NM.values
# Mahakil = df.Mahakil.values
# Rose = df.Rose.values
# data = [CNN, Blsmt, Adasyn, NM, Mahakil, Rose]

# plt.subplot(122)
# plt.boxplot(data)
# plt.xticks(np.arange(1, 7),['CNN', 'Blsmt', 'Adasyn', 'NM', 'Mahakil', 'Rose'])
# plt.xlabel('(b) Chromium')
# plt.ylabel('F1.')
# plt.savefig('box.eps', format='eps', dpi=1000)


df = pd.read_excel('工作簿2.xlsx', sheet_name='Sheet2')
lr = df.lr.values
mnb = df.mnb.values
svm = df.svm.values
mlp = df.mlp.values
rf = df.rf.values
data = [lr, mnb, svm, mlp, rf]

plt.boxplot(data)
plt.xticks(np.arange(1, 6),['LR', 'MNB', 'SVM', 'MLP', 'RF'])
plt.xlabel('Chromium')
plt.ylabel('F1.')
plt.savefig('boxes.eps', format='eps', dpi=1000)