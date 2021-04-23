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


#data_name = 'wicket'
imb_approach = 'Rose' # Farsec, CNN, ROSE, Mahakil
data_names =['OpenStack'] #,'Ambari','Camel','Derby','Wicket', 'Chromium', 'OpenStack'


for data_name in data_names:
    output = '../output/0_ordered/es/'+ data_name + '_' +imb_approach+ '_10times.csv'      
    csv_file = open(output, "w", newline='')
    writer = csv.writer(csv_file, delimiter=',')
    writer.writerow(['dataname','approach', 'pd', 'prec', 'f1', 'accuracy'])

    train_csv = '../input/ordered/'+ data_name +'.csv'
    print('*********Begin:',data_name)
    data = pandas.read_csv(train_csv).fillna('')
    data_content = data.description
    data_label = data.security
    
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
      
#    counter = Counter(train_label)
#    counter_pos = counter.get(0)
#    counter_neg = counter.get(1)#    
#    average = int((counter_pos + counter_neg)/ 2) 
    
    for i in range(0,10):   
#        pipe = make_pipeline(
#                RandomUnderSampler(sampling_strategy={0:average},random_state=42),
#                RandomOverSampler(sampling_strategy={1:average},random_state=42),
#                )
#    #    print(train_label)
#        train_content_matrix_imb, train_label_imb = pipe.fit_resample(train_content_matrix, train_label)

    #    train_content_matrix, train_label = imbalance_strategies.get_ovs_BorderlineSMOTE(train_content_matrix, train_label)  
#        train_content_matrix_imb, train_label_imb = Mahala(train_content_matrix, train_label)
        cnn = CondensedNearestNeighbour(random_state=42)
        train_content_matrix_imb, train_label_imb = cnn.fit_resample(train_content_matrix, train_label) 
    
        clf = RandomForestClassifier(oob_score=True, n_estimators=30)
        clf.fit(train_content_matrix, train_label)
        predicted = clf.predict(test_content_matrix.toarray())
    
        TP, FN, TN, FP, pd, prec, f1, accuracy\
            = mf.model_measure_mop(predicted, test_label)
    #        print(TP, FN, TN, FP, pd,  prec, f_measure, success_rate)
    #        print()
        writer.writerow([data_name, imb_approach, pd, prec, f1, accuracy])
    csv_file.close()
print(output + '**************** finished************************')

