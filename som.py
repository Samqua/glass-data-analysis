# -*- coding: utf-8 -*-
"""
Created on Tue Jun 26 16:51:43 2018

@author: Samqua
"""

import numpy as np
import math
import datetime
import random

class som:
    """
    Version 1.0, June 26th 2018
    
    Each instance of som is a self-organizing map,
    with methods for training on a new data set
    as well as classification/clusering
    (returning the indices of the closest node/neuron,
    given a vector in the input space).
    Initialize an instance with an array of weight vectors
    which function as the initial coordinate vectors
    for each node in the map. It is recommended that
    these weights be chosen randomly, rather than
    from a subspace of PCA eigenvectors.
    The dimensions of this array of weight vectors
    define the shape of the self organizing map,
    in essence the "resolution" of the whole process,
    although each vector in the array must have the
    same dimension as the vectors in the input data space.
    """
    
    # variables that every SOM has
    # max_iterations=5 # now a property of the training algorithm
    threshold=0.85 # fraction of training data to discard at random each training iteration
    # learning_rate=0.01 # learning rate has been replaced by a variable function of the step s
    
    def __init__(self, name, initial_weights):
        self.name=name
        self.initial=initial_weights
        self.ydim=len(initial_weights)
        self.xdim=len(initial_weights[0])
        self.vdim=len(initial_weights[0][0])
        self.network=initial_weights
    
    def learning_rate(self,s,itermax):
        """
        Learning rate as a function of the iteration s.
        """
        # n.b. average distance in 7000-dimensional space is ~30; in this data set it appears to be 50.
        if s<10:
            return 2.5
        else:
            return 2.5*math.exp(-s/(itermax/8)) # arbitrary; but the prefactor could cause numerical runaway if much greater than 2.
    
    def distance(self,v1,v2):
        """
        Euclidean distance metric, for now.
        Percentage speed increase of numpy functions
        over Python loops is on the order of 25000%.
        """
        return math.sqrt(np.sum(np.square(v1-v2)))
    
    def neighborhood(self,i,j,i0,j0,s):
        """
        Unnormalized gaussian neighborhood function.
        Determines the "influence" neuron (i0,j0)
        has on neuron (i,j) at step s.
        """
        return math.exp(-((i-i0)**2 + (j-j0)**2)/(((0.35*(self.ydim))/(s+0.25))**2)) # the 0.35 is arbitrary, as is the choice of ydim over xdim
    
    def vonNeumann(self,i0,j0,r):
        """
        *** currently unused ***
        Returns a list of indices of the von Neumann neighbors
        of neuron (i0,j0) within Manhattan distance r.
        Implemented as a naive search over the entire network---
        with Python loops no less---this could stand to be improved.
        """
        neighbors=[]
        for i in range(self.ydim):
            for j in range(self.xdim):
                if abs(i-i0)+abs(j-j0)<=r:
                    if i==i0 and j==j0:
                        continue
                    else:
                        neighbors.append((i,j))
        return neighbors
    
    def bmu(self,data_vector,quiet=True):
        """
        Returns the SOM indices (row, column)
        of the node closest to the input data_vector,
        called the 'best matching unit' (BMU).
        Employed in the training method, although it may
        also be used for classification after training.
        """
        distances=np.apply_along_axis(lambda x:self.distance(x,data_vector),2,self.network)
        if quiet is False:
            print("Distance to BMU: "+str(distances.flat[np.argmin(distances)]))
        return np.unravel_index(distances.argmin(), distances.shape)
        """
        #start=datetime.datetime.now()
        best=math.inf
        best_coords=(0,0)
        for i in range(self.ydim):
            for j in range(self.xdim):
                dist=self.distance(data_vector,self.network[i][j])
                if dist<best:
                    best=dist
                    best_coords=(i,j)
        #print("TOTAL RUN TIME: "+str(datetime.datetime.now()-start))
        if quiet is False:
            print("Distance from BMU: "+str(best))
        return best_coords
        """
    
    def train(self,training_data,max_iterations):
        start=datetime.datetime.now()
        print("Start: "+str(start))
        print("Log10 threshold score per iteration per n per vdim per xdim per ydim: "+str(math.log((self.threshold)/(max_iterations*len(training_data)*len(training_data[0])*self.xdim*self.ydim),10)))
        for s in range(max_iterations):
            for td_index in range(len(training_data)):
                q=random.random()
                if q<self.threshold: # discard some percentage of the training data each iteration
                    continue
                else:
                    print("Training "+str(self.name)+", iteration "+str(s+1)+" of "+str(max_iterations)+"...")
                    print("Current observation: "+str(td_index))
                    current_bmu=self.bmu(training_data[td_index],quiet=False)
                    for i in range(self.ydim):
                        for j in range(self.xdim):
                            # update formula
                            # note that this could be rewritten as a numpy broadcast with the unfortunate exception of the neighborhood function which depends on the indices (i,j) of the weight vector to be updated
                            self.network[i,j]=self.network[i,j]+self.neighborhood(i,j,current_bmu[0],current_bmu[1],s)*(self.learning_rate(s,max_iterations))*(training_data[td_index]-self.network[i,j])
        np.save(str(self.name)+" "+datetime.datetime.now().strftime("%y-%m-%d-%H-%M-%S"),self.network) # saves current SOM as a human-unreadable .npy file
        print("Total run time: "+str(datetime.datetime.now()-start))
        print("Seconds per iteration: "+str((datetime.datetime.now()-start)/max_iterations))
    
    def dimreduce(self,data_matrix,validation=False):
        """
        Maps high-dimensional data matrix to the
        current two-dimensional manifold defined by
        self.network. Apply this after training
        unless you want garbage.
        """
        lowdrep=np.apply_along_axis(self.bmu,1,data_matrix)
        if validation is True:
            np.savetxt(str(self.name)+"-dimreduce-validation.csv",lowdrep,delimiter=';',fmt='%i')
        else:
            np.savetxt(str(self.name)+"-dimreduce.csv",lowdrep,delimiter=';',fmt='%i')
        return lowdrep

"""
glass_train_data_kernel=np.genfromtxt('glass_train_data_kernel.csv', delimiter=';', dtype=float)

soms=[]
for i in range(9): # try 9 a night
    soms.append(som("som"+str(i),np.random.random((100,100,7000))))

beginning=datetime.datetime.now()

for x in soms:
    x.train(glass_train_data_kernel,65) # this will take a while...

print("Complete runtime: "+str(datetime.datetime.now()-beginning))
"""