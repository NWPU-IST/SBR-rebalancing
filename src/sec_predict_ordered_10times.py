# -*- coding: utf-8 -*-
"""
Created on Fri Oct 16 22:18:56 2020

@author: xingyu
"""

import csv
import time
import math
from scipy import sparse

import pandas 
import numpy as np
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

from sklearn.utils import shuffle

import dimension_reduce as dr
import model_measure_functions as mf
import imbalance_strategies
from MAHAKIL import Mahala
from imblearn.under_sampling import EditedNearestNeighbours, NearMiss
from imblearn.under_sampling import CondensedNearestNeighbour
from imblearn.ensemble import BalancedBaggingClassifier
from imblearn.ensemble import EasyEnsembleClassifier
from imblearn.pipeline import Pipeline, make_pipeline
from imblearn.under_sampling import TomekLinks,RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler, SVMSMOTE, SMOTE, ADASYN
from sklearn.metrics import brier_score_loss

import farsec_text_filter as farsec
#from sklearn.metrics import mean_squared_error as mse


#data_name = 'wicket'
imb_approach = 'Rose' # Farsec, CNN, ROSE, Mahakil
data_names =['Ambari'] #,'Ambari','Camel','Derby','Wicket', 'Chromium', 'OpenStack'


for data_name in data_names:
    output = '../output/0_ordered/es/'+ data_name + '_' +imb_approach+ '_10times.csv'      
    csv_file = open(output, "w", newline='')
    writer = csv.writer(csv_file, delimiter=',')
    writer.writerow(['dataname','approach', 'pd', 'prec', 'f1', 'accuracy', 'MCC', 'AUC', 'brier'])

    train_csv = '../input/ordered/'+ data_name +'.csv'
    print('*********Begin:',data_name)
    
    
    for i in range(0,10):  
        data = pandas.read_csv(train_csv).fillna('')
        data_content = data.description
        data_label = data.security
        
        num_h = int(1/2*len(data_label))

        train_content = data_content[:num_h]
        train_label = data_label[:num_h]
       
        test_content = data_content[num_h:]
        test_label = data_label[num_h:]
        test_label = test_label.tolist()
        
        
        # train_content,train_label = farsec.FARSEC(train_content, train_label, 50, 0.1)
        
        vectorizer = CountVectorizer(stop_words='english')
        train_content_matrix = vectorizer.fit_transform(train_content)
        test_content_matrix = vectorizer.transform(test_content)
        
        train_content_matrix, test_content_matrix \
            = dr.selectFromLinearSVC2(train_content_matrix,train_label,test_content_matrix)  
          
        counter = Counter(train_label)
        counter_pos = counter.get(0)
        counter_neg = counter.get(1)#    
        average = int((counter_pos + counter_neg)/ 2)     
        pipe = make_pipeline(
                RandomUnderSampler(sampling_strategy={0:average},random_state=42),
                RandomOverSampler(sampling_strategy={1:average},random_state=42),
                )
#    #    print(train_label)
        #train_content_matrix_imb, train_label_imb = pipe.fit_resample(train_content_matrix, train_label)

    #    train_content_matrix, train_label = imbalance_strategies.get_ovs_BorderlineSMOTE(train_content_matrix, train_label)  
        train_content_matrix_imb, train_label_imb = Mahala(train_content_matrix, train_label)
        #cnn = CondensedNearestNeighbour(random_state=42)
        #train_content_matrix_imb, train_label_imb = cnn.fit_resample(train_content_matrix, train_label) 
    
        #clf = LogisticRegression()
        #clf = MultinomialNB()
        #clf = svm.SVC(kernel='linear', probability=False)
        #clf = MLPClassifier(max_iter=5000,shuffle = True)
        clf = RandomForestClassifier(oob_score=True, n_estimators=30)
        clf.fit(train_content_matrix, train_label)
        predicted = clf.predict(test_content_matrix.toarray())
        predicted_prob = clf.predict_proba(test_content_matrix.toarray())
        #predicted_prob = clf.decision_function(test_content_matrix.toarray())
        predicted_prob = predicted_prob[:, 1]
        auc = roc_auc_score(test_label, predicted_prob)
        #auc = 0
        #losses = np.subtract(test_label, predicted_prob)**2
        # 该行上下两行的代码与birer_xcore_loss的计算结果完全相同
        #brier = losses.sum() / (len(test_label))
        brier = brier_score_loss(test_label, predicted_prob)
        #brier = 0
        TP, FN, TN, FP, pd, prec, f1, accuracy\
            = mf.model_measure_mop(predicted, test_label)
    #        print(TP, FN, TN, FP, pd,  prec, f_measure, success_rate)
    #        print()
        mcc = (TP*TN-FP*FN)/math.sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))
        writer.writerow([data_name, imb_approach, pd, prec, f1, accuracy, mcc, auc, brier])
    csv_file.close()
print(output + '\n**************** finished************************')

