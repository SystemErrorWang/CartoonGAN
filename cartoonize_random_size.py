import cv2
import tensorflow as tf
import numpy as np
import tensorflow.contrib.slim as slim

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

def resblock(inputs, out_channel=32, name='resblock'):
    
    with tf.variable_scope(name):
        
        x = slim.convolution2d(inputs, out_channel, [3, 3], 
                               activation_fn=None, scope='conv1')
        x = tf.nn.leaky_relu(x)
        x = slim.convolution2d(x, out_channel, [3, 3], 
                               activation_fn=None, scope='conv2')
        
        return x + inputs
    
    

def network(inputs, channel=32, num_blocks=4, name='generator', reuse=False):
    with tf.variable_scope(name, reuse=reuse):
        
        x = slim.convolution2d(inputs, channel, [7, 7], activation_fn=None)
        x = tf.nn.leaky_relu(x)
        
        x = slim.convolution2d(x, channel*2, [3, 3], stride=2, activation_fn=None)
        x = slim.convolution2d(x, channel*2, [3, 3], activation_fn=None)
        x = tf.nn.leaky_relu(x)
       
        x = slim.convolution2d(x, channel*4, [3, 3], stride=2, activation_fn=None)
        x = slim.convolution2d(x, channel*4, [3, 3], activation_fn=None)
        x = tf.nn.leaky_relu(x)
        
        for idx in range(num_blocks):
            x = resblock(x, out_channel=channel*4, name='block_{}'.format(idx))
 
        x = slim.conv2d_transpose(x, channel*2, [3, 3], stride=2, activation_fn=None)
        x = slim.convolution2d(x, channel*2, [3, 3], activation_fn=None)
        x = tf.nn.leaky_relu(x)
        
        x = slim.conv2d_transpose(x, channel, [3, 3], stride=2, activation_fn=None)
        x = slim.convolution2d(x, channel, [3, 3], activation_fn=None)
        x = tf.nn.leaky_relu(x)
        
        x = slim.convolution2d(x, 3, [7, 7], activation_fn=None)
        x = tf.clip_by_value(x, -1, 1)
        
        return x
    
    


def cartoonize(image_path, weight_path):
    inputs = tf.placeholder(tf.float32, [1, None, None, 3])
    output = network(inputs)
    all_vars = tf.trainable_variables()
    gene_vars = [var for var in all_vars if 'generator' in var.name]
    
    sess = tf.Session()
    weight = np.load(weight_path)
    for param, var in zip(weight, gene_vars):
        sess.run(var.assign(param))
    image = cv2.imread(image_path)
    image = image.astype(np.float32)/127.5-1
    image = np.expand_dims(image, axis=0)
    cartoon_image = sess.run(output, {inputs: image})
    cartoon_image = np.squeeze(cartoon_image)
    cartoon_image = (cartoon_image+1)*127.5
    cartoon_image = np.clip(cartoon_image, 0, 255)
    cartoon_image = cartoon_image.astype(np.uint8)
    return cartoon_image

            
if __name__ == '__main__':
    

    image_path = 'kyoto1.jpg'
    weight_path = 'cartoon_weight.npy'
    cartoon_image = cartoonize(image_path, weight_path)
    cv2.imwrite('cartoon.jpg', cartoon_image)