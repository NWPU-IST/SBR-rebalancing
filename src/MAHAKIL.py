# -*- coding: utf-8 -*-
"""
Created on Tue Sep 17 10:09:08 2019

@author: xingyu
"""

import scipy as sp
from scipy.spatial.distance import mahalanobis
import pandas 

from sklearn.utils import shuffle

def Mahala(train_content_matrix, train_label):
    df = pandas.DataFrame(train_content_matrix.toarray())
    length = df.shape[1]
    row_number = [i for i,v in enumerate(train_label)if v == 0]
    df_neg = df.loc[row_number, :]
    label_neg = [0 for i in range(len(row_number))]
    df.drop(row_number, axis=0, inplace=True)
    col_number = [i for i,v in df.iteritems() if any(v)==False]
    df.drop(col_number, axis=1, inplace=True)
    #df = df - df.mean(axis=0)
    covmx = df.cov()  # 计算协方差矩阵
    try:
        invcovmx = sp.linalg.inv(covmx)  # 计算方阵的逆矩阵
    except:
        invcovmx = covmx
    rows, cols = df.shape
    df['vectors']=df.iloc[:,[i for i in range(cols)]].values.tolist()
    tmp = df.iloc[:,[i for i in range(cols)]].mean(axis=0).values.tolist()
    df['mean'] = [tmp for i in range(rows)]
    df['mahala'] = df.apply(lambda x:mahalanobis(x['vectors'],x['mean'],invcovmx),axis=1)
    df.fillna(0, inplace=True)
    df.sort_values(by='mahala', inplace=True, ascending=False)
    
    
    df_pre = df.iloc[0:rows//2, [i for i in range(cols)]]
    df_last = df.iloc[rows//2:,[i for i in range(cols)]]
    df_pre.reset_index(drop=True, inplace=True)
    df_last.reset_index(drop=True, inplace=True)
    df_new = (df_pre+df_last)/2
    df_new.dropna(axis=0, how='any',inplace=True)
    
    x_chk = [df_pre, df_last, df_new]
    x_new = [[df_pre, df_last, df_new]]
    x_tmp = []
    while True:
        l = sum([v.shape[0] for v in x_chk])
        if l > len(row_number):
            break
        for pre, last, new in x_new:
            t = []
            tmp = (pre + new) / 2
            tmp.dropna(axis=0, how='any', inplace = True)
            t.append([pre, new, tmp])
            x_chk.append(tmp)
            tmp = (last + new) / 2
            tmp.dropna(axis=0, how='any', inplace = True)
            t.append([last,new,tmp])
            x_chk.append(tmp)
            x_tmp.extend(t)
            
        x_new = x_tmp
        x_tmp = []
    
    df = x_chk[0]
    for v in x_chk[1:]:
        df = df.append(v)
        
    df.reset_index(drop=True, inplace=True)
    
    df = df.reindex(columns=[i for i in range(length)], fill_value = 0)
    label_pos = [1 for i in range(df.shape[0])]
    label_train = label_pos+label_neg
    df = df.append(df_neg)
    df['label'] = label_train
    df = shuffle(df)
    train_label = df['label'].values.tolist()
    df.drop(['label'], axis=1, inplace=True)
    train_content_matrix = df.values
    
    return train_content_matrix, train_label