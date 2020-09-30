import os

import numpy as np
import pandas as pd
import tensorflow as tf
from tf.keras.callbacks import EarlyStopping
from tf.keras.layers.advanced_activations import PReLU, ReLu
from tf.keras.layers.core import Activation, Dense, Dropout
from tf.keras.layers.normalization import BatchNormalization
from tf.keras.models import Sequential, load_model
from tf.keras.utils import np_utils
from tf.keras.optimizer import SGD, Adam

from sklearn.preprocessing import StandardScaler

from model import Model
from util import Util

class ModelNN(Model):
    def train(self, tr_x, tr_y, va_x=None, va_y=None):
        
        validation = va_x is not None
        
        # Standardize
        self.scaler = StandardScaler()
        self.scaler.fit(tr_x)
        tr_x = self.scaler.transform(tr_x)
        
        if validation:
            va_x = scaler.transform(va_x)
        
        input_dropout = self.params['input_dropout'] # float: ration of unit droped out in input layer
        hidden_layers = self.params['hidden_layers'] # int: number of hidden layer
        hidden_units = self.params['units'] # int: number of units for each hidden layer
        hidden_activation = self.params['hidden_activation'] # str(prelu, relu) : activation function of hidden layer
        hidden_dropout = self.params['hidden_dropout'] # int: number of units droped out in each hidden layer
        batch_norm = self.params['batch_norm'] # str (before_act): whether perfom BatchNormalization before activation
        optimizer_type = self.params['optimizer']['type'] # str (sgd, adam): optimizer method 
        optimizer_lr = self.params['optimizer']['lr'] # float: learning_rate of optimizer
        batch_size = self.params['batch_size'] # int: size of each batch

        self.model = Sequential()

        # Input Layer
        self.model.add(Dropout(input_dropout, input_shape=(tr_x.shape[1], )))
        
        # Hidden Layers
        for i in range(hidden_layers):
            self.model.add(Dense(hidden_units))
            if batch_norm == 'before_act':
                self.model.add(BatchNormalization())
            
            if hidden_activation == 'prelu':
                self.model.add(PReLU)
            elif hidden_activation == 'reru':
                self.model.add(ReLu)
            else:
                raise NotImplementedError

            self.model.add(Dropout(hidden_dropout))

        # Output Layer
        self.model.add(Dense(1))

        # Optimizer
        if optimizer_type == 'sgd':
            optimizer = SGD(lr=optimizer_lr, decay=1e-6, momentum=0.9, neterov=True)
        elif optimizer_type == 'adam':
            optimizer = Adam(lr=optimizer_lr, beta_1=0.9, beta_2=0.999, decay=0.)
        else:
            raise NotImplementedError
            
        # Loss and Evaluation
        self.model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['mse'])

        # Number of Epoch, Early Stropping
        nb_epoch = 200
        patience = 20
        early_stopping = EarlyStopping(patience=patience, restore_best_weights=True)

        if validation:
            self.model.fit(tr_x, tr_y, 
                           epochs=nb_epoch, 
                           batch_size=batch_size, verbose=1, 
                           validation_data=(va_x, va_y), 
                           callbacks=[early_stopping])
        else:
            self.model.fit(tr_x, tr_y, 
                           epochs=nb_epoch, 
                           batch_size=batch_size, verbose=1)

        
    def predict(self, te_x):

        te_x = self.scaler.transform(te_x)
        y_pred = self.model.predict(te_x)
        return y_pred

            

