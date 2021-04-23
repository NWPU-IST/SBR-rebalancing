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
#from sklearn.metrics import mean_squared_error as mse


#data_name = 'wicket'

data_names =['OpenStack'] #,'Ambari','Camel','Derby','Wicket','openstack'


for data_name in data_names:
    output = '../output/ordered/'+data_name+'_10folds_rose_output.csv'      
    csv_file = open(output, "w", newline='')
    writer = csv.writer(csv_file, delimiter=',')
    writer.writerow(['dataname', 'clf', 'TP', 'FN', 'TN', 'FP', 'pd', 'prec', 'f1', 'accuracy'])
    train_csv = '../input/ordered/'+ data_name +'.csv'
    
#    col_names = ['content', 'label']
    data = pandas.read_csv(train_csv).fillna('')
#    train_data = shuffle(train_data)
    data_content = data.content
    data_label = data.sec
    
    num0 = int(1/11*len(data_label))
    for i in range(0,10):
        train_content = data_content[i*num0:(i+1)*num0]
        train_label = data_label[i*num0:(i+1)*num0]
       
        test_content = data_content[(i+1)*num0:(i+2)*num0]
        test_label = data_label[(i+1)*num0:(i+2)*num0]
        test_label = test_label.tolist()
        
        vectorizer = CountVectorizer(stop_words='english')
        clf_names = [ 'lr', 'nb', 'mlp' , 'svm','rf'] #'lr', 'nb', 'mlp' , 'svm','rf'
        
        train_content_matrix = vectorizer.fit_transform(train_content)
        test_content_matrix = vectorizer.transform(test_content)
        
    #    counter = Counter(train_label)
    #    counter_pos = counter.get(0)
    #    counter_neg = counter.get(1)
        
        average = len(test_label)    
        
        train_content_matrix, test_content_matrix \
            = dr.selectFromLinearSVC2(train_content_matrix,train_label,test_content_matrix)  
    
#        print('******************',average)
        pipe = make_pipeline(
                RandomUnderSampler(sampling_strategy={0:average},random_state=42),
                RandomOverSampler(sampling_strategy={1:average},random_state=42),
                )
    #    
    #    train_content_matrix, train_label = pipe.fit_resample(train_content_matrix, train_label)
        #pipe.fit(train_content_matrix, train_label)
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
    #    train_content_matrix, train_label = cnn.fit_resample(train_content_matrix_dr, train_label)
        
    #    clf = EasyEnsembleClassifier()
    #    clf.fit(train_content_matrix, train_label)
    #    predicted = clf.predict(test_content_matrix)
    #    TP, FN, TN, FP, pd, f_measure, g_measure, success_rate, auc \
    #            = mf.model_measure_mop(predicted, test_label)
    #    print(TP, FN, TN, FP, pd, f_measure,
    #                         g_measure, success_rate, auc)
    #    print()
        
        for clf_name in clf_names:
            if clf_name == 'lr':
                clf = LogisticRegression()
            elif clf_name == 'svm':
                # the kernel can also be 'linear', 'rbf','polynomial','sigmoid', etc.
                clf = svm.SVC(kernel='linear', probability=True)
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
    #        print('****************************** predicted:',predicted.tolist())
    #        print('******************************:test label:',test_label)
    #        predicted_proba = clf.predict_proba(test_content_matrix)
    #        TP, FN, TN, FP, pd, pf, prec, f_measure, g_measure, success_rate, auc, PofB20, opt \
    #            = mf.model_measure_with_cross(predicted, predicted_proba, test_label)
    #        print('++++++++++++++++++++++++++\n', predicted)
    #        print('++++++++++++++++++++++++++test_label:\n', test_label)       
            TP, FN, TN, FP, pd, prec, f_measure, success_rate\
                = mf.model_measure_mop(predicted, test_label)
            print(TP, FN, TN, FP, pd,  prec, f_measure, success_rate)
    #        print()
            writer.writerow([data_name, clf_name.upper(), TP, FN, TN, FP, pd, prec, f_measure, success_rate])

    csv_file.close()
print(output + '**************** finished************************')

