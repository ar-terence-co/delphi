# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function

import tensorflow as tf

tf.enable_eager_execution()

class ResnetLayerBlock1D(tf.keras.Model):
    def __init__(self, kernel_size, filters):
        super(ResnetLayerBlock1D, self).__init__(name='')
        filters1, filters2, filters3 = filters
        
        self.conv1a = tf.keras.layers.Conv1D(filters1, 1)
        self.bn1a = tf.keras.layers.BatchNormalization()
        
        self.conv1b = tf.keras.layers.Conv1D(filters2, kernel_size, padding='same')
        self.bn1b = tf.keras.layers.BatchNormalization()
        
        self.conv1c = tf.keras.layers.Conv1D(filters3, 1)
        self.bn1c = tf.keras.layers.BatchNormalization()
        
    def call(self, input_tensor, training=False):
        x = self.conv1a(input_tensor)
        x = self.bn1a(x, training=training)
        x = tf.nn.relu(x)
        
        x = self.conv1b(input_tensor)
        x = self.bn1b(x, training=training)
        x = tf.nn.relu(x)
        
        x = self.conv1c(input_tensor)
        x = self.bn1c(x, training=training)
        
        x = x + input_tensor
        x = tf.nn.relu(x)
        
        return x

class network_delta():
    def __init__(self, input_shape, checkpoint_dir):
        sequential = []
        
        sequential.append(tf.keras.layers.Conv1D(64, 7, strides=2, padding='same', input_shape=input_shape))
        
        #Resnet
        kernel_size = 3
        
        filters = 64 #[64, 64, 256]
        for i in range(3):
            sequential.append(ResnetLayerBlock1D(kernel_size, [filters, filters, 4*filters]))
            
        sequential.append(tf.keras.layers.MaxPooling1D(pool_size=2, strides=2))

        filters = 128 #[128, 128, 512]
        for i in range(4):
            sequential.append(ResnetLayerBlock1D(kernel_size, [filters, filters, 4*filters]))

        sequential.append(tf.keras.layers.MaxPooling1D(pool_size=2, strides=2))

        filters = 256 #[256, 256, 1024]
        for i in range(6):
            sequential.append(ResnetLayerBlock1D(kernel_size, [filters, filters, 4*filters]))
            
        sequential.append(tf.keras.layers.MaxPooling1D(pool_size=2, strides=2))

        filters = 512 #[512, 512, 2048]
        for i in range(3):
            sequential.append(ResnetLayerBlock1D(kernel_size, [filters, filters, 4*filters]))
            
        sequential.append(tf.keras.layers.AveragePooling1D(pool_size=2, strides=2))
        
        sequential.append(tf.keras.layers.Dense(1024, activation=tf.nn.relu))
        sequential.append(tf.keras.layers.Dense(1))
        
        self.model = tf.keras.Sequential(sequential)
        
        self.optimizer = tf.train.AdamOptimizer()
        
        self.global_step = tf.Variable(0)
        
        self.checkpoint_dir = checkpoint_dir
        
        
    def predict(self, images, threshold=0.5):
        logits = self.model(images)
        probabilities= tf.nn.sigmoid(logits)
        predictions = tf.to_float(tf.greater_equal(probabilities,threshold))
        print("Probabilities: {}".format(probabilities))
        print("Classes: {}".format(predictions))
        return {
            'classes': predictions,
            'probabilities': probabilities
        }
        
    def loss(self, images, labels):
        logits = self.model(images)
        return tf.losses.sigmoid_cross_entropy(labels, logits=logits)
    
    def grad(self, images, labels):
        with tf.GradientTape() as tape:
            loss_value = self.loss(images, labels)
        return loss_value, tape.gradient(loss_value, self.model.trainable_variables)
    
    def train(self, images, labels, epochs=1, batch_size=32, threshold=0.5, save=True):
        train_loss_results = []
        train_accuracy_results = []
        train_precision_results = []
        train_recall_results = []
        local_step = 0
        
        if len(images) % batch_size > 0:
            raise ValueError('Input should be divisible by the batch size')
        if len(images) != len(labels):
            raise ValueError('Input and labels should be of the same length')
        
        for epoch in range(epochs):
            epoch_loss_avg = tf.metrics.mean
            epoch_accuracy = tf.metrics.accuracy
            epoch_precision = tf.metrics.precision
            epoch_recall = tf.metrics.recall
            
            for i in range(0,len(images),batch_size):
                x = images[i:i+batch_size]
                y = images[i:i+batch_size]
                print(x)
                print(y)
                loss_value, grads = self.loss(x, y)
                self.optimizer.apply_gradients(zip(grads, self.model.variables), self.global_step)
                local_step += 1
                
                epoch_loss_avg(loss_value)
                
                predictions = self.predict(x, threshold)
                epoch_accuracy(labels, predictions['classes'])
                epoch_precision(labels, predictions['classes'])
                epoch_recall(labels, predictions['classes'])
            
            train_loss_results.append(epoch_loss_avg.result())
            train_accuracy_results.append(epoch_accuracy.result())
            train_precision_results.append(epoch_precision.result())
            train_recall_results.append(epoch_recall.result())

            print("Epoch {:03d}: Loss: {:.3f}, Accuracy: {:.3%}, Precision: {:.3%}, Recall: {:.3%}"
                  .format(epoch, 
                          epoch_loss_avg.result(), 
                          epoch_accuracy.result(), 
                          epoch_precision.result(), 
                          epoch_recall.result()
                  )
            )
            
            if save:
                self.save_model()
                print("Model Checkpoint Saved!")
        
        return {
                'loss': train_loss_results, 
                'accuracy': train_accuracy_results, 
                'precision': train_precision_results, 
                'recall': train_recall_results
        }
    
    def validate(self, images, labels, threshold=0.5):
        loss_avg = tf.metrics.mean
        accuracy = tf.metrics.accuracy
        precision = tf.metrics.precision
        recall = tf.metrics.recall
        
        loss_value, grads = self.loss(images, labels)
        
        loss_avg(loss_value)
        
        predictions = self.predict(images, threshold)
        accuracy(labels, predictions['classes'])
        precision(labels, predictions['classes'])
        recall(labels, predictions['classes'])

        print("Loss: {:.3f}, Accuracy: {:.3%}, Precision: {:.3%}, Recall: {:.3%}"
              .format(loss_avg.result(), 
                      accuracy.result(), 
                      precision.result(), 
                      recall.result()
              )
        )
        
        return {
                'loss': loss_avg.result(), 
                'accuracy': accuracy.result(), 
                'precision': precision.result(), 
                'recall': recall.result()
        }  
        
    def save_model(self):
        self.model.save_weights(self.checkpoint_dir)
        
    def load_model(self):
        self.model.load_weights(self.checkpoint_dir)

            
            
    
    
        
        
        
    
    