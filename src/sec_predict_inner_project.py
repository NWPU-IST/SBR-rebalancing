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

data_names =['Ambari'] #,'camel','derby','openstack'


for data_name in data_names:
#    output = '../output/train_test/'+data_name+'_base_output.csv'      
#    csv_file = open(output, "w", newline='')
#    writer = csv.writer(csv_file, delimiter=',')
    train_csv = '../input/ordered/'+ data_name +'.csv'
    test_csv = '../input/ordered/'+ data_name +'.csv'
    #cand_csv = '../input/3_3_4/'+ data_name +'_cand.csv'
    
    col_names = ['content', 'label']
    train_data = pandas.read_csv(train_csv, names=col_names, header=None).fillna('')
#    train_data = shuffle(train_data)
    train_content = train_data.content
    train_label = train_data.label
    
    #cand_data = pandas.read_csv(cand_csv, names=col_names, header=None).fillna('')
    #train_data = np.vstack((train_data,cand_data))
    #shuffle(train_data)
    #train_content = train_data[:,0]
    #train_label = train_data[:,1].tolist()
    
    test_data = pandas.read_csv(test_csv, names=col_names, header=None).fillna('')
    test_content = test_data.content
    test_label = test_data.label
    
    vectorizer = CountVectorizer(stop_words='english')
    clf_names = [ 'lr', 'nb', 'mlp' , 'svm','rf'] #'lr', 'nb', 'mlp' , 'svm','rf'
    
    train_content_matrix = vectorizer.fit_transform(train_content)
    test_content_matrix = vectorizer.transform(test_content)
    
    
    train_content_matrix, test_content_matrix \
        = dr.selectFromLinearSVC2(train_content_matrix,train_label,test_content_matrix)  
    counter = Counter(train_label)
    counter_pos = counter.get(0)
    counter_neg = counter.get(1)
    average = (counter_pos + counter_neg) // 2
    print('******************',average)
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
            clf = MLPClassifier(solver='lbfgs', alpha=1e-5,
                                hidden_layer_sizes=(5, 2), random_state=1)
        elif clf_name == 'nb':
            clf = MultinomialNB()
        elif clf_name == 'rf':
            clf = RandomForestClassifier(oob_score=True, n_estimators=30)
        else:
            print('分类器名称仅为\'lr,svm,mlp,nb,rf\'中的一种')
    
        clf.fit(train_content_matrix, train_label)
    #        print(train_content_matrix_input_dmr.shape)
        predicted = clf.predict(test_content_matrix.toarray())
#        predicted_proba = clf.predict_proba(test_content_matrix)
#        TP, FN, TN, FP, pd, pf, prec, f_measure, g_measure, success_rate, auc, PofB20, opt \
#            = mf.model_measure_with_cross(predicted, predicted_proba, test_label)
        TP, FN, TN, FP, pd, prec, f_measure, g_measure, success_rate, auc \
            = mf.model_measure_mop(predicted, test_label)
        print(TP, FN, TN, FP, pd,  prec, f_measure,
                         success_rate)
#        print()
#        writer.writerow([data_name, clf_name, TP, FN, TN, FP, pd, pf, prec, f_measure,
#                         g_measure, success_rate, auc, PofB20, opt])

#    csv_file.close()
#print(output + '**************** finished************************')

