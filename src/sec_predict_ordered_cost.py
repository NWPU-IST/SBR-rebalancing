# coding: utf-8
import csv
import time
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

from sklearn.utils import shuffle

import dimension_reduce as dr
import model_measure_functions as mf
import imbalance_strategies
from MAHAKIL import Mahala
from imblearn.under_sampling import EditedNearestNeighbours
from imblearn.under_sampling import CondensedNearestNeighbour
from imblearn.ensemble import BalancedBaggingClassifier
from imblearn.ensemble import EasyEnsembleClassifier
from imblearn.pipeline import Pipeline, make_pipeline
from imblearn.under_sampling import TomekLinks,RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler, SVMSMOTE, SMOTE

import farsec_text_filter as farsec
#from sklearn.metrics import mean_squared_error as mse

from datetime import datetime  
  
 
#data_name = 'wicket'
imb_approach = 'Rose' # # Base, Farsec, Smotebl,Adasyn, NM, CNN, ROSE, Mahala
data_names =['OpenStack'] #,'Ambari','Camel','Derby','Wicket', 'Chromium', 'OpenStack'

output = '../output/0_ordered/'+ imb_approach+ '_OpenStack_cost1.csv'      
csv_file = open(output, "w", newline='')
writer = csv.writer(csv_file, delimiter=',')
writer.writerow(['dataname','approach', 'TP', 'FN', 'TN', 'FP', 'pd', 'prec', 'f1', 'accuracy','second','microsecond'])
print('*********Begin:',output)
for data_name in data_names:
    start_ = datetime.utcnow() 
    train_csv = '../input/ordered/'+ data_name +'.csv'
    print('*********Begin:',data_name)
#    col_names = ['content', 'label']
    data = pandas.read_csv(train_csv).fillna('')
#    train_data = shuffle(train_data)
    data_content = data.description
    data_label = data.security
#    print(data_label)
    
    num_h = int(1/2*len(data_label))

    train_content = data_content[:num_h]
    train_label = data_label[:num_h]
   
    test_content = data_content[num_h:]
    test_label = data_label[num_h:]
    test_label = test_label.tolist()
    
    
#    train_content,train_label = farsec.FARSEC(train_content, train_label, 50, 0.1)
    
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
#    print(train_label)
    train_content_matrix, train_label = pipe.fit_resample(train_content_matrix, train_label)
    pipe.fit(train_content_matrix, train_label)
#    train_content_matrix, train_label = imbalance_strategies.get_ovs_smote_standard(train_content_matrix, train_label)
#    train_content_matrix, train_label = imbalance_strategies.get_ovs_adasyn(train_content_matrix, train_label)  
#    train_content_matrix, train_label = imbalance_strategies.get_ovs_BorderlineSMOTE(train_content_matrix, train_label)  
#    train_content_matrix, train_label = imbalance_strategies.get_uds_rdm(train_content_matrix, train_label) 
#    train_content_matrix, train_label = imbalance_strategies.get_uds_nm(train_content_matrix, train_label) 
#    train_content_matrix, train_label = imbalance_strategies.get_uds_CNN(train_content_matrix, train_label) 
#    train_content_matrix, train_label = imbalance_strategies.get_ens_BalanceCascade(train_content_matrix, train_label) 
#    train_content_matrix, train_label = imbalance_strategies.get_ens_EasyEnsemble(train_content_matrix, train_label)
#    train_content_matrix, train_label = imbalance_strategies.get_uds_enn(train_content_matrix, train_label)  #Nonetype occured
#    train_content_matrix, train_label = imbalance_strategies.get_cbs_smoteenn(train_content_matrix, train_label)  #Nonetype occured
#    train_content_matrix, train_label = imbalance_strategies.get_cbs_smotetomek(train_content_matrix, train_label)  #Nonetype occured
#    train_content_matrix, train_label = Mahala(train_content_matrix, train_label)
#    enn = EditedNearestNeighbours() 
#    train_content_matrix, train_label = enn.fit_resample(train_content_matrix_dr, train_label)    
#    cnn = CondensedNearestNeighbour(random_state=42)
#    train_content_matrix, train_label = cnn.fit_resample(train_content_matrix, train_label)

    clf_names = ['rf'] #'lr', 'nb', 'mlp' , 'svm','rf'
  
    for clf_name in clf_names:
        if clf_name == 'lr':
            clf = LogisticRegression()
        elif clf_name == 'svm':
            # the kernel can also be 'linear', 'rbf','polynomial','sigmoid', etc.
            clf = svm.SVC(kernel='linear', probability=False)
    #        clf = svm.SVC(kernel='rbf', probability = True, class_weight = 1, decision_function_shaope = 'ovr')
        elif clf_name == 'mlp':
            clf = MLPClassifier(max_iter=5000,shuffle = True)
        elif clf_name == 'nb':
            clf = MultinomialNB()
        elif clf_name == 'rf':
            clf = RandomForestClassifier(oob_score=True, n_estimators=30)
        else:
            print('分类器名称仅为\'lr,svm,mlp,nb,rf\'中的一种')
    
        clf.fit(train_content_matrix, train_label)
    #        print(train_content_matrix_input_dmr.shape)
        predicted = clf.predict(test_content_matrix.toarray())

        TP, FN, TN, FP, pd, prec, f_measure, success_rate\
            = mf.model_measure_mop(predicted, test_label)
        print(TP, FN, TN, FP, pd,  prec, f_measure, success_rate)
#        print()

        end_ = datetime.utcnow()  
        c = (end_ - start_)  
        writer.writerow([data_name, imb_approach +'+'+ clf_name.upper(), TP, FN, TN, FP, pd, prec, f_measure, success_rate, c.seconds, c.microseconds])
#        print('**Time Cost:', data_name, c.seconds)    
#        print(c.microseconds)
csv_file.close()
print(output + '**************** finished************************')

