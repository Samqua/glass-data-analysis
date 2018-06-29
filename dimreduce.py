# -*- coding: utf-8 -*-
"""
Created on Thu Jun 28 13:15:35 2018

@author: Samqua
"""

import numpy as np
from som import som
import datetime

##############################################################
# This file is for performing batch dimreduce after training
# Ignore this if you performed dimreduce during training
##############################################################

glass_validation_data_kernel=np.genfromtxt('glass_validation_data_kernel.csv', delimiter=';', dtype=float)

for i in range(14):
    print("Current i: "+str(i))
    print(datetime.datetime.now())
    current_som=som("som"+str(i),np.load("som"+str(i)+".npy"))
    current_som.dimreduce(glass_validation_data_kernel,validation=True)
    print(datetime.datetime.now())