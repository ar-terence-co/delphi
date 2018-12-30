# -*- coding: utf-8 -*-

import tensorflow as tf

# Residual Neural Network based on https://arxiv.org/pdf/1512.03385.pdf
def network_delta(features, labels, mode, params):    
    #(IN: ?, 128, 4)
    input_layer = tf.to_float(features['x'])
    input_layer_size = params['shape'][0]
    
    #Input filter (IN: ?, 128, 4 OUT: ?, 128, 64)
    filters_in = 64
    conv_in = tf.layers.conv1d(
        inputs=input_layer,
        filters=filters_in,
        kernel_size=5,
        strides=1,
        padding='same',
        use_bias=False,
        activation=None        
    )
    bn_in = tf.layers.batch_normalization(
        inputs=conv_in,
        axis=-1,
        training=mode == tf.estimator.ModeKeys.TRAIN
    )
    conv_in = tf.nn.relu(bn_in)
    conv_in_size = input_layer_size
    
    #RESNET Block 1 64, 64, 256 (IN; ?, 128, 64 OUT: ?, 128, 256)
    filters_1 = 64 
    conv_1 = conv_in
    for i in range(3):
        conv_1 = tf.layers.conv1d(
            inputs=conv_1,
            filters=filters_1,
            kernel_size=1,
            padding='same',
            use_bias=False,
            activation=None
        )
        bn_1 = tf.layers.batch_normalization(
            inputs=conv_1,
            axis=-1,
            training=mode == tf.estimator.ModeKeys.TRAIN
        )
        conv_1 = tf.nn.relu(bn_1) 
        
        conv_1 = tf.layers.conv1d(
            inputs=conv_1,
            filters=filters_1,
            kernel_size=3,
            padding='same',
            use_bias=False,
            activation=None
        )
        bn_1 = tf.layers.batch_normalization(
            inputs=conv_1,
            axis=-1,
            training=mode == tf.estimator.ModeKeys.TRAIN
        )
        conv_1 = tf.nn.relu(bn_1) 
        
        conv_1 = tf.layers.conv1d(
            inputs=conv_1,
            filters=4*filters_1,
            kernel_size=1,
            padding='same',
            use_bias=False,
            activation=None
        )
        bn_1 = tf.layers.batch_normalization(
            inputs=conv_1,
            axis=-1,
            training=mode == tf.estimator.ModeKeys.TRAIN
        )
        if i == 0:
            ch = (4*filters_1 - filters_in) / 2
            padded_conv_in = tf.pad(conv_in, [[0,0],[0,0],[ch, ch]])
            bn_1 = tf.add(bn_1, padded_conv_in)
        conv_1 = tf.nn.relu(bn_1) 
        
    conv_1_size = conv_in_size
    
    #Max Pooling 1 (IN: ?, 128, 256 OUT: ?, 64, 256)
    pool_1 = tf.layers.max_pooling1d(
            inputs=conv_1, 
            pool_size=2, 
            strides=2
    )
    pool_1_size = conv_1_size / 2

    #RESNET Block 2 128, 128, 512 (IN; ?, 64, 256 OUT: ?, 64, 512)
    filters_2 = 128 
    conv_2 = pool_1
    for i in range(3):
        conv_2 = tf.layers.conv1d(
            inputs=conv_2,
            filters=filters_2,
            kernel_size=1,
            padding='same',
            use_bias=False,
            activation=None
        )
        bn_2 = tf.layers.batch_normalization(
            inputs=conv_2,
            axis=-1,
            training=mode == tf.estimator.ModeKeys.TRAIN
        )
        conv_2 = tf.nn.relu(bn_2) 
        
        conv_2 = tf.layers.conv1d(
            inputs=conv_2,
            filters=filters_2,
            kernel_size=3,
            padding='same',
            use_bias=False,
            activation=None
        )
        bn_2 = tf.layers.batch_normalization(
            inputs=conv_2,
            axis=-1,
            training=mode == tf.estimator.ModeKeys.TRAIN
        )
        conv_2 = tf.nn.relu(bn_2) 
        
        conv_2 = tf.layers.conv1d(
            inputs=conv_2,
            filters=4*filters_2,
            kernel_size=1,
            padding='same',
            use_bias=False,
            activation=None
        )
        bn_2 = tf.layers.batch_normalization(
            inputs=conv_2,
            axis=-1,
            training=mode == tf.estimator.ModeKeys.TRAIN
        )
        if i == 0:
            ch = 2 * (filters_2 - filters_1)
            padded_pool_1 = tf.pad(pool_1, [[0,0],[0,0],[ch, ch]])
            bn_2 = tf.add(bn_2, padded_pool_1)
        conv_2 = tf.nn.relu(bn_2) 
        
    conv_2_size = pool_1_size
    
    #Max Pooling 2 (IN: ?, 64, 512 OUT: ?, 32, 512)
    pool_2 = tf.layers.max_pooling1d(
            inputs=conv_2, 
            pool_size=2, 
            strides=2
    )
    pool_2_size = conv_2_size / 2
    
    #RESNET Block 3 256, 256, 1024 (IN; ?, 32, 512 OUT: ?, 32, 1024)
    filters_3 = 256 
    conv_3 = pool_2
    for i in range(3):
        conv_3 = tf.layers.conv1d(
            inputs=conv_3,
            filters=filters_3,
            kernel_size=1,
            padding='same',
            use_bias=False,
            activation=None
        )
        bn_3 = tf.layers.batch_normalization(
            inputs=conv_3,
            axis=-1,
            training=mode == tf.estimator.ModeKeys.TRAIN
        )
        conv_3 = tf.nn.relu(bn_3) 
        
        conv_3 = tf.layers.conv1d(
            inputs=conv_3,
            filters=filters_3,
            kernel_size=3,
            padding='same',
            use_bias=False,
            activation=None
        )
        bn_3 = tf.layers.batch_normalization(
            inputs=conv_3,
            axis=-1,
            training=mode == tf.estimator.ModeKeys.TRAIN
        )
        conv_3 = tf.nn.relu(bn_3) 
        
        conv_3 = tf.layers.conv1d(
            inputs=conv_3,
            filters=4*filters_3,
            kernel_size=1,
            padding='same',
            use_bias=False,
            activation=None
        )
        bn_3 = tf.layers.batch_normalization(
            inputs=conv_3,
            axis=-1,
            training=mode == tf.estimator.ModeKeys.TRAIN
        )
        if i == 0:
            ch = 2 * (filters_3 - filters_2)
            padded_pool_2 = tf.pad(pool_2, [[0,0],[0,0],[ch, ch]])
            bn_3 = tf.add(bn_3, padded_pool_2)
        conv_3 = tf.nn.relu(bn_3) 
        
    conv_3_size = pool_2_size
    
    #Max Pooling 4 (IN: ?, 32, 1024 OUT: ?, 16, 1024)
    pool_3 = tf.layers.max_pooling1d(
            inputs=conv_3, 
            pool_size=2, 
            strides=2
    )
    pool_3_size = conv_3_size / 2    
    
    #RESNET Block 4 512, 512, 2048 (IN; ?, 16, 1024 OUT: ?, 16, 2048)
    filters_4 = 512 
    conv_4 = pool_3
    for i in range(3):
        conv_4 = tf.layers.conv1d(
            inputs=conv_4,
            filters=filters_4,
            kernel_size=1,
            padding='same',
            use_bias=False,
            activation=None
        )
        bn_4 = tf.layers.batch_normalization(
            inputs=conv_4,
            axis=-1,
            training=mode == tf.estimator.ModeKeys.TRAIN
        )
        conv_4 = tf.nn.relu(bn_4) 
        
        conv_4 = tf.layers.conv1d(
            inputs=conv_4,
            filters=filters_4,
            kernel_size=3,
            padding='same',
            use_bias=False,
            activation=None
        )
        bn_4 = tf.layers.batch_normalization(
            inputs=conv_4,
            axis=-1,
            training=mode == tf.estimator.ModeKeys.TRAIN
        )
        conv_4 = tf.nn.relu(bn_4) 
        
        conv_4 = tf.layers.conv1d(
            inputs=conv_4,
            filters=4*filters_4,
            kernel_size=1,
            padding='same',
            use_bias=False,
            activation=None
        )
        bn_4 = tf.layers.batch_normalization(
            inputs=conv_4,
            axis=-1,
            training=mode == tf.estimator.ModeKeys.TRAIN
        )
        if i == 0:
            ch = 2 * (filters_4 - filters_3)
            padded_pool_3 = tf.pad(pool_3, [[0,0],[0,0],[ch, ch]])
            bn_4 = tf.add(bn_4, padded_pool_3)
        conv_4 = tf.nn.relu(bn_4) 
        
    conv_4_size = pool_3_size
    
    #Average Pooling 4 (IN: ?, 16, 2048 OUT: ?, 8, 2048)
    pool_4 = tf.layers.average_pooling1d(
            inputs=conv_4, 
            pool_size=2, 
            strides=2
    )
    pool_4_size = conv_4_size / 2  
    
    #Flatten for Fully Connected Layers (IN: ?, 8, 2048 OUT: ?, 16384)
    
    flatten = tf.reshape(pool_4, (-1, pool_4_size * 4 * filters_4))
    
    #Fully Connected Layer 1 (IN: ?, 16384 OUT: ?, 1000)
    dense_fc = tf.layers.dense(
            inputs=flatten,
            units=1024,
            use_bias=False,
            activation=None
    )
    bn_fc = tf.layers.batch_normalization(
        inputs=dense_fc,
        axis=-1,
        training=mode == tf.estimator.ModeKeys.TRAIN
    )
    dense_fc = tf.nn.relu(bn_fc)    
    
    #Logits (IN: ?, 1000 OUT: ?, 1)
    logits=tf.layers.dense(
            inputs=dense_fc,
            units=1
    )
    
    probabilities = tf.nn.sigmoid(logits, name='probabilities_tensor')
    
    #Prediction
    predictions={
        'probabilities': probabilities,
        'classes_t0.3': tf.to_float(tf.greater_equal(probabilities, tf.constant(0.3))),
        'classes_t0.5': tf.to_float(tf.greater_equal(probabilities, tf.constant(0.5))),
        'classes_t0.7': tf.to_float(tf.greater_equal(probabilities, tf.constant(0.7)))
    }
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)
    
    #Training
    loss = tf.losses.sigmoid_cross_entropy(labels, logits=logits)
    
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.AdamOptimizer()
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_op = optimizer.minimize(
                loss=loss,
                global_step=tf.train.get_global_step()
            )
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)
    
    #Evaluation
    eval_metric_ops = {
        'accuracy_t0.3': tf.metrics.accuracy(labels=labels, predictions=predictions['classes_t0.3']),
        'precision_t0.3': tf.metrics.precision(labels=labels, predictions=predictions['classes_t0.3']),
        'recall_t0.3': tf.metrics.recall(labels=labels, predictions=predictions['classes_t0.3']),  
        'accuracy_t0.5': tf.metrics.accuracy(labels=labels, predictions=predictions['classes_t0.5']),
        'precision_t0.5': tf.metrics.precision(labels=labels, predictions=predictions['classes_t0.5']),
        'recall_t0.5': tf.metrics.recall(labels=labels, predictions=predictions['classes_t0.5']),
        'accuracy_t0.7': tf.metrics.accuracy(labels=labels, predictions=predictions['classes_t0.7']),
        'precision_t0.7': tf.metrics.precision(labels=labels, predictions=predictions['classes_t0.7']),
        'recall_t0.7': tf.metrics.recall(labels=labels, predictions=predictions['classes_t0.7']),
    }
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)