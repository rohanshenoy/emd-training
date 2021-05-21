# -*- coding: utf-8 -*-
"""
Draft code for implementation of EMD with a CNN model to train ASIC Autoencoder

Takes in a pair of numbers and then outputs their loss

There are 8 .h5 files with optimized EMD, load all of them, architecture is currently 443, look into this later

"""

import numpy as np
import pandas as pd
import math

import itertools

import os
import sys
sys.path.insert(0, "../")

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from plotWafer import plotWafer
from train import emd

from tensorflow.keras.layers import Input, Dense, Flatten, Concatenate, BatchNormalization, Activation, Average, Conv2D
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l1_l2
        
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

import tensorflow as tf
from tensorflow.keras import backend as K

use443=True
def __init__(self,use443):
    self.use443=use443
        
arrange443 = np.array([0,16, 32,
                       1,17, 33,
                       2,18, 34,
                       3,19, 35,
                       4,20, 36,
                       5,21, 37,
                       6,22, 38,
                       7,23, 39,
                       8,24, 40,
                       9,25, 41,
                       10,26, 42,
                       11,27, 43,
                       12,28, 44,
                       13,29, 45,
                       14,30, 46,
                       15,31, 47])

"""

remap_8x8 = [4, 12, 20, 28,  5, 13, 21, 29,  6, 14, 22, 30,  7, 15, 23, 31, 
              24, 25, 26, 27, 16, 17, 18, 19,  8,  9, 10, 11,  0,  1,  2,  3, 
              59, 51, 43, 35, 58, 50, 42, 34, 57, 49, 41, 33, 56, 48, 40, 32]

remap_8x8_matrix = np.zeros(48*64,dtype=np.float32).reshape((64,48))


    
for i in range(48): 
    remap_8x8_matrix[remap_8x8[i],i] = 1
    
"""

def emd_loss(y_true,y_pred):
    
    #y_true = K.cast(y_true, y_pred.dtype)
    
    """
    Input comes in in 8x8 looking like this:
    arrange8x8 = np.array([
        28,29,30,31,0,4,8,12,
        24,25,26,27,1,5,9,13,
        20,21,22,23,2,6,10,14,
        16,17,18,19,3,7,11,15,
        47,43,39,35,35,34,33,32,
        46,42,38,34,39,38,37,36,
        45,41,37,33,43,42,41,40,
        44,40,36,32,47,46,45,44])
    Remapping using array from telescope.py
    
    """
    print(tf.shape(y_true).numpy())
    remap_8x8 = [4, 12, 20, 28,  5, 13, 21, 29,  6, 14, 22, 30,  7, 15, 23, 31, 
              24, 25, 26, 27, 16, 17, 18, 19,  8,  9, 10, 11,  0,  1,  2,  3, 
              59, 51, 43, 35, 58, 50, 42, 34, 57, 49, 41, 33, 56, 48, 40, 32]
    
    

    remap_8x8_matrix = np.zeros(48*64,dtype=np.float32).reshape((64,48))

    for i in range(48): 
        remap_8x8_matrix[remap_8x8[i],i] = 1
        
    
    
    #y_true=K.reshape((y_true,(-1,64)),remap_8x8_matrix)
    #y_pred=K.reshape((y_pred,(-1,64)),remap_8x8_matrix)
    
    y_pred_443 = (y_pred)[:,remap_8x8_matrix].reshape(-1,8,8,1)
    y_true_443 = (y_true)[:,remap_8x8_matrix].reshape(-1,8,8,1)
    
    
    #CNN EMD Reshape using arrange 443
    y_pred_443 = (y)[:,arrange443].reshape(-1,4,4,3)
    y_true_443 = (y)[:,arrange443].reshape(-1,4,4,3)
    
    X1_train = y_pred_443
    X1_val = y_true_443
        
    #Loading EMD Models with num_model.h5 [1,8]
    
    model_directory=os.path.join(current_directory,r'Best/3.h5')
    print(model_directory)

    input1 = Input(shape=(4, 4, 3,), name='input_1')
    input2 = Input(shape=(4, 4, 3,), name='input_2')
    x = Concatenate(name='concat')([input1, input2])

    output = Dense(1, name='output')(x)

    model = load_model(model_directory)
    model.summary()

    # make a model that enforces the symmetry of the EMD function by averging the outputs for swapped inputs

    output = Average(name='average')([model((input1, input2)), model((input2, input1))])
    sym_model = Model(inputs=[input1, input2], outputs=output, name='sym_model')
    sym_model.summary()
   
    sym_model.compile(optimizer='adam', loss='msle', metrics=['mse', 'mae', 'mape', 'msle'])
    history = sym_model.fit((X1_train, X2_train), y_train, validation_data=((X1_val, X2_val), y_val), epochs=num_epochs, verbose=1, batch_size=32, callbacks=callbacks)
        
    y_val_preds = sym_model.predict((X1_val, X2_val))
    (y_val_preds[y_val>0].flatten()-y_val[y_val>0].flatten())/y_val[y_val>0].flatten()
    
    rel_diff = (y_val_preds[y_val>0].flatten()-y_val[y_val>0].flatten())/y_val[y_val>0].flatten()
       
    return(np.std(rel_diff))
