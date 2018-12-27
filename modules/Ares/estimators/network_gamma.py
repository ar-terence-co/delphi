# -*- coding: utf-8 -*-

import tensorflow as tf

# Plain Convolutional Neural Network based on https://arxiv.org/pdf/1512.03385.pdf
def network_gamma(features, labels, mode, params):    
    input_layer = tf.to_float(features['x'])
    input_layer_size = params['shape'][0]
    
    #Convolution 1 128 dim and 64 channels x 4
    conv1 = input_layer
    for i in range(6):
        conv1 = tf.layers.conv1d(
            inputs=conv1,
            filters=64,
            kernel_size=[3],
            padding='same',
            use_bias=False,
            activation=None
        )
        bn1 = tf.layers.batch_normalization(
            inputs=conv1,
            axis=-1,
            momentum=0.9,
            epsilon=0.01,
            center=True,
            scale=True,
            training=mode == tf.estimator.ModeKeys.TRAIN,
        )
        conv1 = tf.nn.relu(bn1)        
    conv1_size = input_layer_size

    #Max Pooling 1 64 dim
    pool1 = tf.layers.max_pooling1d(
            inputs=conv1, 
            pool_size=[2], 
            strides=2
    )
    pool1_size = conv1_size/2
    
    #Convolution 2 64 dim and 128 channels x 4
    conv2 = pool1
    for i in range(8):
        conv2 = tf.layers.conv1d(
            inputs=conv2,
            filters=128,
            kernel_size=[3],
            padding='same',
            use_bias=False,
            activation=None,
            name='conv2_'+str(i)+'_tensor'
        )
        print(conv2)
        bn2 = tf.layers.batch_normalization(
            inputs=conv2,
            axis=-1,
            momentum=0.9,
            epsilon=0.01,
            center=True,
            scale=True,
            training=mode == tf.estimator.ModeKeys.TRAIN,
        )
        conv2 = tf.nn.relu(bn2, name='conv2_bn_'+str(i)+'_tensor')    
    conv2_size = pool1_size
    
    #Max Pooling  2 32 dim
    pool2 = tf.layers.max_pooling1d(
            inputs=conv2, 
            pool_size=[2], 
            strides=2
    )
    pool2_size = conv2_size/2
    
    #Convolution 3 32 dim and 256 channels x 4
    conv3 = pool2
    for i in range(12):
        conv3 = tf.layers.conv1d(
            inputs=conv3,
            filters=256,
            kernel_size=[3],
            padding='same',
            use_bias=False,
            activation=None
        )
        bn3 = tf.layers.batch_normalization(
            inputs=conv3,
            axis=-1,
            momentum=0.9,
            epsilon=0.01,
            center=True,
            scale=True,
            training=mode == tf.estimator.ModeKeys.TRAIN,
        )
        conv3 = tf.nn.relu(bn3)    
    conv3_size = pool2_size
    
    #Max Pooling 3 16 dim
    pool3 = tf.layers.max_pooling1d(
            inputs=conv3, 
            pool_size=[2], 
            strides=2
    )
    pool3_size = conv3_size/2
    
    #Convolution 4 16 dim and 512 channels x 4
    conv4 = pool3
    for i in range(6):
        conv4 = tf.layers.conv1d(
            inputs=conv4,
            filters=512,
            kernel_size=[3],
            padding='same',
            use_bias=False,
            activation=None,
        )
        bn4 = tf.layers.batch_normalization(
            inputs=conv4,
            axis=-1,
            momentum=0.9,
            epsilon=0.01,
            center=True,
            scale=True,
            training=mode == tf.estimator.ModeKeys.TRAIN,
        )
        conv4 = tf.nn.relu(bn4)    
    conv4_size = pool3_size
    
    #Max Pooling 4 8 dim
    pool4 = tf.layers.max_pooling1d(
            inputs=conv4, 
            pool_size=[2], 
            strides=2
    )
    pool4_size = conv4_size/2
    
    flattened_conv = tf.reshape(pool4, [-1, pool4_size * 512])
    
    #Fully Connected Layer
    dense1 = tf.layers.dense(
            inputs=flattened_conv,
            units=2048,
            use_bias=False,
            activation=None
    )    
    bn_dense1 = tf.layers.batch_normalization(
            inputs=dense1,
            axis=-1,
            momentum=0.9,
            epsilon=0.01,
            center=True,
            scale=True,
            training=mode == tf.estimator.ModeKeys.TRAIN,
    )
    dense1 = tf.nn.relu(bn_dense1)
    
    dense2 = tf.layers.dense(
            inputs=dense1,
            units=1024,
            use_bias=False,
            activation=None
    )
    bn_dense2 = tf.layers.batch_normalization(
            inputs=dense2,
            axis=-1,
            momentum=0.9,
            epsilon=0.01,
            center=True,
            scale=True,
            training=mode == tf.estimator.ModeKeys.TRAIN,
    )
    dense2 = tf.nn.relu(bn_dense2)  

    #Logits
    logits=tf.layers.dense(
            inputs=dense2,
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
        'precision': tf.metrics.precision(labels=labels, predictions=predictions['classes']),
        'recall': tf.metrics.recall(labels=labels, predictions=predictions['classes']),     
    }
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)