# -*- coding: utf-8 -*-
"""
Created on Sat Jun 23 14:05:11 2018

@author: Samqua
"""

import numpy as np
import matplotlib.pyplot as plt
import datetime

#glass_data=np.genfromtxt('glass_train_data.csv', delimiter=';', dtype=float)
#glass_data_star=np.genfromtxt('glass_train_data_star.csv', delimiter=';', dtype=float)
glass_train_data_kernel=np.genfromtxt('glass_train_data_kernel.csv', delimiter=';', dtype=float)
#glass_train_data_kernel_star=np.genfromtxt('glass_train_data_kernel_star.csv', delimiter=';', dtype=float)
glass_labels=np.genfromtxt('glass_train_labels.csv', delimiter=';', dtype=int)

v1=2492
v2=2493
mobility=0

plt.figure(figsize=(10,10))
plt.title("Mobility "+str(mobility))
plt.xlabel("V"+str(v1))
plt.ylabel("V"+str(v2))
plt.scatter(glass_train_data_kernel[:,v1],glass_train_data_kernel[:,v2],c=glass_labels[:,mobility],cmap="winter")
plt.savefig('images/PCA scatterplot'+datetime.datetime.now().strftime("%y-%m-%d-%H-%M-%S")+'.png',dpi=300)