# -*- coding: utf-8 -*-
"""
Created on Thu Jun 28 12:50:25 2018

@author: Samqua
"""

import numpy as np
from sklearn import neighbors
from sklearn import metrics
import math


all_preds=[]
mobility=0
glass_validation_labels=np.genfromtxt('glass_validation_labels.csv', delimiter=';', dtype=int)
glass_train_labels=np.genfromtxt('glass_train_labels.csv', delimiter=';', dtype=int)
num_soms=14


for i in range(num_soms):
    somindex=i
    
    som_dimreduce=np.genfromtxt('som'+str(somindex)+'-dimreduce.csv',delimiter=';', dtype=int)
    som_dimreduce_validation=np.genfromtxt('som'+str(somindex)+'-dimreduce-validation.csv',delimiter=';', dtype=int)
    
    clf=neighbors.KNeighborsClassifier(n_neighbors=5,weights='distance')
    clf.fit(som_dimreduce,glass_train_labels[:,mobility])
    
    pred=clf.predict(som_dimreduce_validation)
    print("Accuracy: ")
    print(1-metrics.zero_one_loss(glass_validation_labels[:,mobility],pred))
    all_preds.append(pred)

all_preds=np.array(all_preds)
final_pred=np.sum(all_preds,axis=0)

def criterion(x):
    if x>=math.floor(num_soms/2):
        return 1
    else:
        return 0

criterion=np.vectorize(criterion)
final_pred=criterion(final_pred)
print("Final accuracy: ")
print(1-metrics.zero_one_loss(glass_validation_labels[:,mobility],final_pred))