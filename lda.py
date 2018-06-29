# -*- coding: utf-8 -*-
"""
Created on Mon Jun 25 12:17:45 2018

@author: Samqua
"""

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
import matplotlib.pyplot as plt

glass_train_data=np.genfromtxt('glass_train_data_kernel_star.csv', delimiter=';', dtype=float)
glass_train_labels=np.genfromtxt('glass_train_labels.csv', delimiter=';', dtype=int)[:,3]
glass_validation_data=np.genfromtxt('glass_validation_data_kernel_star.csv', delimiter=';', dtype=float)
glass_validation_labels=np.genfromtxt('glass_validation_labels.csv', delimiter=';', dtype=int)[:,3]

clf=LinearDiscriminantAnalysis()
clf.fit(glass_train_data,glass_train_labels)
#lda_data=clf.transform(glass_train_data)

#rand=np.random.random(size=len(lda_data[:,0]))
#plt.figure()
#plt.scatter(lda_data[:,0],rand,c=glass_train_labels,cmap="winter") # random y coordinate is exclusively for ease of visualization

def zerooneerror(vector1,vector2):
    if len(vector1)!=len(vector2):
        return "lengths aren't the same"
    else:
        z=0
        for i in range(len(vector1)):
            if vector1[i]!=vector2[i]:
                z+=1
        return 1-z/len(vector1)

pred=clf.predict(glass_validation_data)
print(zerooneerror(pred,glass_validation_labels))