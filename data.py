# -*- coding: utf-8 -*-
"""
Created on Thu Jun 21 15:43:53 2018

@author: Samqua
"""

import math
import numpy as np
import scipy.linalg as linalg

# this file is sort of a disaster

def mathematicatize(list):
    return str(list).replace('[','{').replace(']','}')

#================================================================
# - formats data into np arrays of 64-bit floats
# - centers each column (i.e. each variable)
# - performs PCA by eigendecomposition of the covariance matrix
# - partitions data into training data and validation data
# - (1577 and 300 row observations, respectively)
#================================================================

glass_data=np.genfromtxt('glass_data.csv', delimiter=';', skip_header=1,dtype=float) # reads every value, including indices, as float; standard (statistical) form

# time to do some housekeeping
glass_data=glass_data.T
glass_data=np.delete(glass_data,0,0) # deletes indices (which are now garbage floats, not ints) from array
for i in range(len(glass_data)):
    glass_data[i]=glass_data[i]-np.mean(glass_data[i]) # centering
glass_data=glass_data.T
# end housekeeping

# each column is a variable with mean 0, and each row is an observation (the usual statistical form of a data matrix)
#print(glass_data) # standard (statistical) form
#print(len(glass_data),len(glass_data[0])) # num observations, num variables

budget_covariance=np.matmul(glass_data.T,glass_data)
eig=linalg.eigh(budget_covariance)
values=eig[0]
vectors=eig[1] # access the kth smallest eigenvector with vectors[:,k]
values=np.flip(values,axis=0)
vectors=np.flip(vectors,axis=1) # axis must be 1 (!); access the kth largest eigenvector with vectors[:,k] and the first k largest with vectors[:,0:k]
np.savetxt("pca_eigenvalues.csv",values,fmt='%.8e')
np.savetxt("pca_eigenvectors.csv",vectors,delimiter=';',fmt='%.6f')
glass_data_star=np.matmul(glass_data,vectors)[:,0:1875] # the ratio of the variances of principal components i and j is precisely the ratio of their eigenvalues lambda_i/lambda_j
# keep only as many columns (variables) as we have observations
np.savetxt("glass_data_star.csv",glass_data_star,delimiter=';',fmt='%.6f')

#print(glass_data_star)
#print(values)
#print(mathematicatize(glass_data_star[:,0].tolist()))

# take the first 1577 observations and put them into a new CSV for training
glass_train_data_star=glass_data_star[0:1577]
np.savetxt("glass_train_data_star.csv",glass_train_data_star,delimiter=';',fmt='%.6f')
# put the remaining 300 into a validation set
glass_validation_data_star=glass_data_star[1577:1878]
np.savetxt("glass_validation_data_star.csv",glass_validation_data_star,delimiter=';',fmt='%.6f')

glass_labels=np.genfromtxt('glass_labels.csv', delimiter=';', skip_header=1,dtype=int) # reads every value as int
glass_labels=glass_labels.T
glass_labels=np.delete(glass_labels,0,0) # deletes indices
glass_labels=glass_labels.T
glass_train_labels=glass_labels[0:1577]
glass_validation_labels=glass_labels[1577:1878]
np.savetxt("glass_train_labels.csv",glass_train_labels,delimiter=';', fmt='%i')
np.savetxt("glass_validation_labels.csv",glass_validation_labels, delimiter=';', fmt='%i')


#======================================================
#  Maps each triplet of spatial coordinates to a
#  quartet of new coordinates on the surface of 
#  3-sphere embedded in 4D space to account for
#  periodic boundary conditions.
#
#  This effectively converts our 6000-dimensional
#  data set (1000 3-positions and 1000 3-velocities)
#  to a 7000-dimensional data set (1000 4-positions
#  and 1000 3-velocities).
#======================================================

glass_train_data=np.genfromtxt('glass_train_data.csv', delimiter=';', dtype=float)
glass_validation_data=np.genfromtxt('glass_validation_data.csv', delimiter=';', dtype=float)
#glass_train_data_kernel=glass_train_data

glass_train_data_kernel=np.zeros((1577,7000))
glass_validation_data_kernel=np.zeros((300,7000))

def parameterization(x,y,z,r=1,side_length=9.4103602888102831):
    """
    Maps a triplet (x,y,z) in periodic 3-space
    to a quartet (x1,x2,x3,x4) in unperiodic 4-space
    confined to the surface of the 3-sphere.
    The chosen parameterization is:
        x1=r*cos(psi)
        x2=r*sin(psi)*cos(theta)
        x3=r*sin(psi)*sin(theta)*cos(phi)
        x4=r*sin(psi)*sin(theta)*sin(phi)
    where psi and theta range from 0 to pi,
    and phi ranges from 0 to 2*pi.
    """
    psi=math.pi*x/side_length
    theta=math.pi*y/side_length
    phi=2*math.pi*z/side_length
    x1=r*math.cos(psi)
    x2=r*math.sin(psi)*math.cos(theta)
    x3=r*math.sin(psi)*math.sin(theta)*math.cos(phi)
    x4=r*math.sin(psi)*math.sin(theta)*math.sin(phi)
    return x1,x2,x3,x4

def fill(i,j,array): # fills the jth column of the ith row
    if j>=4000:
        return array[i,j-1000]
    else:
        mod=(j%4)
        if mod==0:
            # maps to x1
            return parameterization(array[i,j],array[i,j+1],array[i,j+2])[0]
        elif mod==1:
            # maps to x2
            return parameterization(array[i,j-1],array[i,j],array[i,j+1])[1]
        elif mod==2:
            # maps to x3
            return parameterization(array[i,j-2],array[i,j-1],array[i,j])[2]
        elif mod==3:
            # maps to x4
            return parameterization(array[i,j-3],array[i,j-2],array[i,j-1])[3]

for i in range(len(glass_train_data_kernel)):
    for j in range(len(glass_train_data_kernel[0])):
        glass_train_data_kernel[i,j]=fill(i,j,glass_train_data)

for i in range(len(glass_validation_data_kernel)):
    for j in range(len(glass_validation_data_kernel[0])):
        glass_validation_data_kernel[i,j]=fill(i,j,glass_validation_data)

# time to do some housekeeping
glass_train_data_kernel=glass_train_data_kernel.T
for i in range(len(glass_train_data_kernel)):
    glass_train_data_kernel[i]=glass_train_data_kernel[i]-np.mean(glass_train_data_kernel[i]) # centering
glass_train_data_kernel=glass_train_data_kernel.T

glass_validation_data_kernel=glass_validation_data_kernel.T
for i in range(len(glass_validation_data_kernel)):
    glass_validation_data_kernel[i]=glass_validation_data_kernel[i]-np.mean(glass_validation_data_kernel[i]) # centering
glass_validation_data_kernel=glass_validation_data_kernel.T
# end housekeeping

np.savetxt("glass_train_data_kernel.csv",glass_train_data_kernel,delimiter=';',fmt='%.6f')
np.savetxt("glass_validation_data_kernel.csv",glass_validation_data_kernel,delimiter=';',fmt='%.6f')

budget_covariance=np.matmul(glass_train_data_kernel.T,glass_train_data_kernel)
eig=linalg.eigh(budget_covariance)
values=eig[0]
vectors=eig[1] # access the kth smallest eigenvector with vectors[:,k]
values=np.flip(values,axis=0)
vectors=np.flip(vectors,axis=1) # axis must be 1 (!); access the kth largest eigenvector with vectors[:,k] and the first k largest with vectors[:,0:k]
np.savetxt("pca_eigenvalues.csv",values,fmt='%.8e')
np.savetxt("pca_eigenvectors.csv",vectors,delimiter=';',fmt='%.6f')
glass_train_data_kernel_star=np.matmul(glass_train_data_kernel,vectors)[:,0:1875] # the ratio of the variances of principal components i and j is precisely the ratio of their eigenvalues lambda_i/lambda_j
glass_validation_data_kernel_star=np.matmul(glass_validation_data_kernel,vectors)[:,0:1875] # the ratio of the variances of principal components i and j is precisely the ratio of their eigenvalues lambda_i/lambda_j

# keep only as many columns (variables) as we have observations
np.savetxt("glass_train_data_kernel_star.csv",glass_train_data_kernel_star,delimiter=';',fmt='%.6f')
np.savetxt("glass_validation_data_kernel_star.csv",glass_validation_data_kernel_star,delimiter=';',fmt='%.6f')