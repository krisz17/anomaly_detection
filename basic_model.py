# -*- coding: utf-8 -*-
"""
Created on Wed Nov  9 22:54:03 2022

@author: krzys
"""

import tensorflow as tf
from tensorflow.keras import Model

class BaseModel(Model):
    '''This is the base model class that inherits some sharable functions to the the models for the training and inference.
    This class also inherits from the tensorflow.keras.Model.
    
    '''
    def __init__(self):
        
        super(BaseModel, self).__init__()
        
        self.weight_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.25)
        self.bias_initializer=tf.zeros_initializer()
    
    def init_variables(self):
        '''Initialize the parameters of the neural network. '''
        
        self.W1=tf.compat.v1.get_variable('W1',shape=[29,20], initializer=self.weight_initializer, dtype=tf.float32)
        self.W2=tf.compat.v1.get_variable('W2',shape=[20,8], initializer=self.weight_initializer, dtype=tf.float32)
        self.W3=tf.compat.v1.get_variable('W3',shape=[8,20], initializer=self.weight_initializer, dtype=tf.float32)
        self.W4=tf.compat.v1.get_variable('W3',shape=[20,29], initializer=self.weight_initializer, dtype=tf.float32)
        
        self.b1=tf.compat.v1.get_variable('b1',shape=[20], initializer=self.bias_initializer, dtype=tf.float32)
        self.b2=tf.compat.v1.get_variable('b2',shape=[8], initializer=self.bias_initializer, dtype=tf.float32)
        self.b3=tf.compat.v1.get_variable('b3',shape=[20], initializer=self.bias_initializer, dtype=tf.float32)
    
    
    def forward_propagation(self, x):
        '''Compute the forward pass given the input features x.
        @param x: input features x
        @return prediction: the reconstructed input features x
        '''
           
        with tf.name_scope('feed_forward'):
            
            # First hidden layer
            z1=tf.linalg.matmul(x, self.W1)+self.b1
            a1=tf.nn.relu(z1)
            
            # Second hidden layer
            z2=tf.linalg.matmul(a1,self.W2)+self.b2
            a2=tf.nn.relu(z2)
            
            # Third hidden layer
            z3=tf.linalg.matmul(a2,self.W3)+self.b3
            a3=tf.nn.relu(z3)
            
            prediction=tf.linalg.matmul(a3,self.W4)
            
        return prediction
    
    
    
class AnomalyDetector(BaseModel):
    '''This class represents the class for training of the neural network for anomaly detection. 
    In particular this class is used for training only. The learned weights and biases will be used later
    by the inference model to make the actual anomaly detection in production environment.
    '''
    
    def __init__(self):
        
        super(AnomalyDetector, self).__init__()
        self.init_variables()
        
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        
    
    def compute_loss(self, x_train):  
        '''Compute MSE loss function.
        
        @param x_train: input features
        '''
      
        mse = tf.keras.losses.MeanSquaredError()
        loss = mse(x_train, self.forward_propagation(x_train))
        
        return loss
    

    def train(self, x_train): 
         '''Train the autoencoder.
      
         @parameter x_train: training input features
         '''
         
         # Compute the gradients and apply the gradient descent step
         with tf.GradientTape() as tape:
             gradients = tape.gradient(self.compute_loss(x_train), self.trainable_variables)
             gradient_variables = zip(gradients, self.trainable_variables)
             self.optimizer.apply_gradients(gradient_variables)