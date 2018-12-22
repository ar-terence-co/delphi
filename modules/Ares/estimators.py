# -*- coding: utf-8 -*-

import tensorflow as tf

def cnn_alpha(features, labels, mode, params):    
    input_layer = tf.reshape(tf.to_float(features['x']),[-1,params['shape'][0],params['shape'][1],1])
    conv1 = tf.layers.conv2d(
            inputs=input_layer,
            filters=128,
            kernel_size=[5, params['shape'][1]],
            padding='valid',
            activation=tf.nn.relu
    )
    conv1_size = params['shape'][0] - 5 + 1
    conv1_1d = tf.reshape(conv1,[-1,conv1_size,128])
    pool1 = tf.layers.max_pooling1d(
            inputs=conv1_1d, 
            pool_size=[2], 
            strides=2
    )
    conv2 = tf.layers.conv1d(
            inputs=pool1,
            filters=256,
            kernel_size=[5],
            padding='valid',
            activation=tf.nn.relu
    )
    pool2 = tf.layers.max_pooling1d(
            inputs=conv2, 
            pool_size=[2], 
            strides=2
    )
    pool2_size = ((conv1_size / 4) - ((5 - 1) / 2)) * 256
    pool2_flat = tf.reshape(pool2, [-1, pool2_size])
    dense = tf.layers.dense(
            inputs=pool2_flat,
            units=2048,
            activation=tf.nn.relu
    )
    dense = tf.layers.dense(
            inputs=pool2_flat,
            units=1024,
            activation=tf.nn.relu
    )
    dropout=tf.layers.dropout(
        inputs=dense,
        rate=0.4,
        training=mode == tf.estimator.ModeKeys.TRAIN
    )
    logits=tf.layers.dense(
            inputs=dropout,
            units=1
    )
    probabilities=tf.nn.sigmoid(logits, name='probabilities_tensor')
    predictions={
        'classes': tf.to_float(tf.greater_equal(probabilities,0.5)),
        'probabilities': probabilities
    }
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)
    
    loss = tf.losses.sigmoid_cross_entropy(multi_class_labels=labels, logits=logits)
    
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.AdamOptimizer()
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step()
        )
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)
    
    eval_metric_ops = {
        'accuracy': tf.metrics.accuracy(labels=labels, predictions=predictions['classes'])
    }
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)
    