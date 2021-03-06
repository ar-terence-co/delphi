# -*- coding: utf-8 -*-

import tensorflow as tf

# Runs but poor performance. 
def network_alpha(features, labels, mode, params):    
    input_layer = tf.to_float(features['x'])
    input_layer_size = params['shape'][0]
    
    #Convolution and Max Pooling 1
    conv1 = tf.layers.conv1d(
            inputs=input_layer,
            filters=128,
            kernel_size=[3],
            padding='same',
            activation=tf.nn.relu
    )
    conv1_size = input_layer_size
    pool1 = tf.layers.max_pooling1d(
            inputs=conv1, 
            pool_size=[2], 
            strides=2
    )
    pool1_size = conv1_size/2
    
    #Convolution and Max Pooling 2
    conv2 = tf.layers.conv1d(
            inputs=pool1,
            filters=512,
            kernel_size=[3],
            padding='same',
            activation=tf.nn.relu
    )
    conv2_reduce = tf.layers.conv1d(
            inputs=conv2,
            filters=128,
            kernel_size=[1],
            padding='same',
            activation=tf.nn.relu
    )
    conv2_size = pool1_size
    pool2 = tf.layers.max_pooling1d(
            inputs=conv2_reduce, 
            pool_size=[2], 
            strides=2
    )
    pool2_size = conv2_size/2
    
    #Convolution and Max Pooling 3
    conv3 = tf.layers.conv1d(
            inputs=pool2,
            filters=512,
            kernel_size=[3],
            padding='same',
            activation=tf.nn.relu
    )
    conv3_reduce = tf.layers.conv1d(
            inputs=conv3,
            filters=128,
            kernel_size=[1],
            padding='same',
            activation=tf.nn.relu
    )
    conv3_size = pool2_size
    pool3 = tf.layers.max_pooling1d(
            inputs=conv3_reduce, 
            pool_size=[2], 
            strides=2
    )
    pool3_size = conv3_size/2
    
    flattened_conv = tf.reshape(pool3, [-1, pool3_size * 128])
    
    #Fully Connected Layers with Dropout
    dense1 = tf.layers.dense(
            inputs=flattened_conv,
            units=2048,
            activation=tf.nn.relu
    )
    dropout1 = tf.layers.dropout(
        inputs=dense1,
        rate=0.4,
        training=mode == tf.estimator.ModeKeys.TRAIN
    )
    
    #Logits
    logits=tf.layers.dense(
            inputs=dropout1,
            units=1
    )
    
    #Probabilities
    probabilities = tf.nn.sigmoid(logits, name='probabilities_tensor')
    
    #Prediction
    predictions={
        'classes': tf.to_float(tf.greater_equal(probabilities,0.5)),
        'probabilities': probabilities
    }
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)
    
    #Training
    loss = tf.losses.sigmoid_cross_entropy(multi_class_labels=labels, logits=logits)
    
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.AdamOptimizer()
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step()
        )
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)
    
    #Evaluation
    eval_metric_ops = {
        'accuracy': tf.metrics.accuracy(labels=labels, predictions=predictions['classes']),
        'auc': tf.metrics.auc(labels=labels, predictions=predictions['probabilities']),
        'precision': tf.metrics.precision(labels=labels, predictions=predictions['classes']),
        'recall': tf.metrics.recall(labels=labels, predictions=predictions['classes']),     
    }
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)