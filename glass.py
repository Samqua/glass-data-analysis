# -*- coding: utf-8 -*-
"""
Created on Tue Jun 19 14:31:13 2018

@author: Samqua
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import datetime
import numpy as np
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO)


def main(unused_argv):
    start=datetime.datetime.now()
    print("START: "+str(start))
    
    glass_train_data=np.genfromtxt('glass_train_data_star.csv', delimiter=';', dtype=float)
    glass_train_labels=np.genfromtxt('glass_train_labels.csv', delimiter=';', dtype=int)
    glass_validation_data=np.genfromtxt('glass_validation_data_star.csv', delimiter=';', dtype=float)
    glass_validation_labels=np.genfromtxt('glass_validation_labels.csv', delimiter=';', dtype=int)
    
    mobility_index=0
    
    ############ TAKE FIRST L COMPONENTS ONLY (or don't) ###########
    L=5
    glass_train_data=glass_train_data[:,0:L]
    glass_validation_data=glass_validation_data[:,0:L]
    ################################################################
    
    training_dict={}
    for i in range(len(glass_train_data[0])):                     # note that L=len(glass_train_data[0])
        training_dict["PC{}".format(i+1)]=glass_train_data[:,i]
    validation_dict={}
    for i in range(len(glass_validation_data[0])):
        validation_dict["PC{}".format(i+1)]=glass_validation_data[:,i]
    
    """
    def training_input_fn():
        return training_dict,glass_train_labels[:,0] # all labels are mobilities of one atom, the second index of glass_train_labels - in this case, atom 0

    def validation_input_fn():
        return validation_dict,glass_validation_labels[:,0]
    """
    
    # define input functions
    training_input_fn = tf.estimator.inputs.numpy_input_fn(
            x=training_dict,
            y=glass_train_labels[:,mobility_index],
            batch_size=32,
            num_epochs=None,
            shuffle=True)
    validation_input_fn = tf.estimator.inputs.numpy_input_fn(
            x=validation_dict,
            y=glass_validation_labels[:,mobility_index],
            num_epochs=1,
            shuffle=True)
    

    feat_cols=[tf.feature_column.numeric_column(x) for x in training_dict]
    
    estimator=tf.estimator.DNNClassifier(feature_columns=feat_cols, hidden_units=[500,500,500,220,5],
    activation_fn=tf.nn.softplus,
    optimizer=tf.train.AdamOptimizer(
            learning_rate=0.0025
    ),
    model_dir="/tmp/glass_dnn_classifier_"+str(L)+"_principal_components"+datetime.datetime.now().strftime("%y-%m-%d-%H-%M-%S"))
    
    estimator.train(input_fn=training_input_fn, steps=5000)
    
    eval_result=estimator.evaluate(input_fn=validation_input_fn)
    
    print('\nValidation set accuracy: {accuracy:0.3f}\n'.format(**eval_result))
    print("END: "+str(datetime.datetime.now()))
    print("RUN TIME: "+str(datetime.datetime.now()-start))

if __name__ == "__main__":
    tf.app.run()
