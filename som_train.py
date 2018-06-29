# -*- coding: utf-8 -*-
"""
Created on Wed Jun 27 13:29:14 2018

@author: Samqua
"""

import numpy as np
import datetime
from som import som

glass_train_data_kernel=np.genfromtxt('glass_train_data_kernel.csv', delimiter=';', dtype=float)
glass_validation_data_kernel=np.genfromtxt('glass_validation_data_kernel.csv', delimiter=';', dtype=float)

soms=[]
for i in range(7): # try 7 a night
    soms.append(som("new"+str(i),np.random.random((100,100,7000))))

beginning=datetime.datetime.now()

for x in soms:
    x.train(glass_train_data_kernel,65) # this will take a while...
    x.dimreduce(glass_train_data_kernel)
    x.dimreduce(glass_validation_data_kernel,validation=True)

print("Complete runtime, including twofold dimreduce: "+str(datetime.datetime.now()-beginning))