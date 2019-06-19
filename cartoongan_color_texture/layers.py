import tensorflow as tf
import numpy as np
import tensorflow.contrib.slim as slim



def adaptive_instance_norm(content, style, epsilon=1e-5):

    c_mean, c_var = tf.nn.moments(content, axes=[1, 2], keep_dims=True)
    s_mean, s_var = tf.nn.moments(style, axes=[1, 2], keep_dims=True)
    c_std, s_std = tf.sqrt(c_var + epsilon), tf.sqrt(s_var + epsilon)

    return s_std * (content - c_mean) / c_std + s_mean


def resblock(inputs, out_channel=32, name='resblock'):
    
    with tf.variable_scope(name):
        
        x = slim.convolution2d(inputs, out_channel, [3, 3], activation_fn=None)
        x = tf.nn.relu(x)
        x = slim.convolution2d(x, out_channel, [3, 3], activation_fn=None)
        
        return x + inputs
    

def resblock_in(inputs, out_channel=32, name='resblock'):
    
    with tf.variable_scope(name):
        
        x = slim.convolution2d(inputs, out_channel, [3, 3], activation_fn=None)
        x = tf.contrib.layers.instance_norm(x)
        x = tf.nn.relu(x)
        x = slim.convolution2d(x, out_channel, [3, 3], activation_fn=None)
        x = tf.contrib.layers.instance_norm(x)
        
        return x + inputs
    
    
def resblock_bn(inputs, out_channel=32, is_training=False, name='resblock'):
    
    with tf.variable_scope(name):
        
        x = slim.convolution2d(inputs, out_channel, [3, 3], activation_fn=None)
        x = slim.batch_norm(x, is_training=is_training, center=True, scale=True)
        x = tf.nn.relu(x)
        x = slim.convolution2d(x, out_channel, [3, 3], activation_fn=None)
        x = slim.batch_norm(x, is_training=is_training, center=True, scale=True)
        
        return x + inputs


def pixel_shuffle(X, scale, out_channel, fix=False):
    
    assert int(X.get_shape()[-1]) == (scale ** 2) * out_channel

    if fix:
        bsize = tf.shape(X)[0]
        h, w = X.get_shape().as_list()[1:3]

    else:
        bsize, h, w = tf.shape(X)[0], tf.shape(X)[1], tf.shape(X)[2]

    Xs = tf.split(value=X, num_or_size_splits=scale, axis=3)
    Xr = tf.concat(Xs, 2)
    X = tf.reshape(Xr, (bsize, scale * h, scale * w, out_channel))

    return X