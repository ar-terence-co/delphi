# -*- coding: utf-8 -*-

import tensorflow as tf
from tensorflow import keras

import numpy as np
import matplotlib.pyplot as plt

from modules.Ares.estimators import cnn_alpha

class AresAI():
        
    def __init__(self):
        self.estimator = None
        self.current_model = None
    
    def split_dataset(self, X, Y, split=0.7):
        if len(X) != len(Y):
            raise ValueError('Images and labels must have the same length')
        
        shuffled_indeces = np.arange(0,len(Y))
        np.random.shuffle(shuffled_indeces)
        split = int(np.ceil(len(Y) * 0.7))
        train_indeces = shuffled_indeces[0:split]
        test_indeces = shuffled_indeces[split:len(Y)]
        
        return (X[train_indeces], Y[train_indeces], X[test_indeces], Y[test_indeces])
    
    def setup(self, model_name, image_shape):
        self.estimator = tf.estimator.Estimator(
            model_fn=cnn_alpha,
            model_dir='saved/' + model_name,
            params={
                "shape": image_shape
            }
        )
        self.current_model = model_name
        
    def train(self, train_images, train_labels, epochs=1):
        if self.estimator is None:
            raise ValueError('Estimator not yet set')
        
        train_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={"x": train_images},
            y=train_labels,
            batch_size=100,
            num_epochs=epochs,
            shuffle=True
        )
        
        # tensors_to_log = {'probabilities':'probabilities_tensor'}
        # logging_hook = tf.train.LoggingTensorHook(
        #         tensors=tensors_to_log, every_n_iter=50
        # )
        
        self.estimator.train(
            input_fn=train_input_fn,
            steps=None,
            # hooks=[logging_hook]
        )
        
    def evaluate(self, eval_images, eval_labels):
        if self.estimator is None:
            raise ValueError('Estimator not yet set')
            
        eval_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={"x": eval_images},
            y=eval_labels,
            num_epochs=1,
            shuffle=False
        )
        eval_results = self.estimator.evaluate(
            input_fn=eval_input_fn
        )
        print(eval_results)
        return eval_results
    
    def predict(self, images):
        if self.estimator is None:
            raise ValueError('Estimator not yet set')
        
        predict_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={"x": images},
            num_epochs=1,
            shuffle=False
        )
        
        predictions = self.estimator.predict(
            input_fn=predict_input_fn
        )
        print(predictions)
        return predictions