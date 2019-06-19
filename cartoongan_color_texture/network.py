import layers
import tensorflow as tf
import numpy as np
import tensorflow.contrib.slim as slim

from tqdm import tqdm


def decoder(content, style, channel=32, num_blocks=4, name='decoder'):
    with tf.variable_scope(name):
        feat = layers.adaptive_instance_norm(content, style)
        x = slim.convolution2d(feat, channel*4, [3, 3])
        x = slim.convolution2d(x, channel*4, [3, 3])
        x = slim.conv2d_transpose(x, channel*4, [3, 3], stride=2)
        '''
        x = slim.convolution2d(x, channel*8, [3, 3], activation_fn=None)
        x = tf.nn.relu(layers.pixel_shuffle(x, 2, channel*2))
        '''
        
        x = slim.convolution2d(x, channel*2, [3, 3])
        x = slim.convolution2d(x, channel*2, [3, 3])
        x = slim.conv2d_transpose(x, channel*2, [3, 3], stride=2)
        '''
        x = slim.convolution2d(x, channel*4, [3, 3], activation_fn=None)
        x = tf.nn.relu(layers.pixel_shuffle(x, 2, channel))
        '''
        
        x = slim.convolution2d(x, channel, [3, 3])
        x = slim.convolution2d(x, channel, [3, 3])
        x = slim.conv2d_transpose(x, channel, [3, 3], stride=2)
        '''
        x = slim.convolution2d(x, channel*4, [3, 3], activation_fn=None)
        x = tf.nn.relu(layers.pixel_shuffle(x, 2, channel))
        '''
        
        x = slim.convolution2d(x, 3, [3, 3], activation_fn=None)
        x = tf.clip_by_value(x, -1, 1)
        return x
        


def generator(inputs, channel=32, num_blocks=4, name='generator', reuse=False):
    with tf.variable_scope(name, reuse=reuse):
        
        x = slim.convolution2d(inputs, channel, [7, 7], activation_fn=None)
        x = tf.nn.relu(x)
        
        x = slim.convolution2d(x, channel*2, [3, 3], stride=2, activation_fn=None)
        x = slim.convolution2d(x, channel*2, [3, 3], activation_fn=None)
        x = tf.nn.relu(x)
       
        x = slim.convolution2d(x, channel*4, [3, 3], stride=2, activation_fn=None)
        x = slim.convolution2d(x, channel*4, [3, 3], activation_fn=None)
        x = tf.nn.relu(x)
        
        for idx in range(num_blocks):
            x = layers.resblock(x, out_channel=channel*4, name='block_{}'.format(idx))
            
        '''
        h, w = tf.shape(x)[1], tf.shape(x)[2]
        x = tf.image.resize_images(x, (h*2, w*2), method=tf.image.ResizeMethod.BILINEAR)
        x = slim.convolution2d(x, channel*2, [3, 3], activation_fn=None)
        '''
        x = slim.convolution2d(x, channel*8, [3, 3], activation_fn=None)
        x = layers.pixel_shuffle(x, 2, channel*2)
        x = slim.convolution2d(x, channel*2, [3, 3], activation_fn=None)
        
        x = tf.nn.relu(x)
        
        '''
        h, w = tf.shape(x)[1], tf.shape(x)[2]
        x = tf.image.resize_images(x, (h*2, w*2), method=tf.image.ResizeMethod.BILINEAR)
        x = slim.convolution2d(x, channel, [3, 3], activation_fn=None)
        '''
        x = slim.convolution2d(x, channel*4, [3, 3], activation_fn=None)
        x = layers.pixel_shuffle(x, 2, channel)
        x = slim.convolution2d(x, channel, [3, 3], activation_fn=None)
        x = tf.nn.relu(x)
        
        x = slim.convolution2d(x, 3, [7, 7], activation_fn=None)
        x = tf.clip_by_value(x, -1, 1)
        
        return x
    
    
    

def generator_in(inputs, channel=32, num_blocks=4, name='generator'):
    with tf.variable_scope(name):
        
        x = slim.convolution2d(inputs, channel, [7, 7], activation_fn=None)
        x = tf.contrib.layers.instance_norm(x)
        x = tf.nn.relu(x)
        
        x = slim.convolution2d(x, channel*2, [3, 3], stride=2, activation_fn=None)
        x = slim.convolution2d(x, channel*2, [3, 3], activation_fn=None)
        x = tf.contrib.layers.instance_norm(x)
        x = tf.nn.relu(x)
       
        x = slim.convolution2d(x, channel*4, [3, 3], stride=2, activation_fn=None)
        x = slim.convolution2d(x, channel*4, [3, 3], activation_fn=None)
        x = tf.contrib.layers.instance_norm(x)
        x = tf.nn.relu(x)
        
        
        for idx in range(num_blocks):
            x = layers.resblock_in(x, out_channel=channel*4, name='block_{}'.format(idx))
            
            
        '''
        x = slim.convolution2d(x, channel*8, [3, 3], activation_fn=None)
        x = layers.pixel_shuffle(x, 2, channel*2)
        x = slim.convolution2d(x, channel*2, [3, 3], activation_fn=None)
        
        x = slim.conv2d_transpose(x, channel*2, [3, 3], stride=2, activation_fn=None)
        x = slim.conv2d_transpose(x, channel*2, [3, 3], activation_fn=None)
        '''
        h, w = tf.shape(x)[1], tf.shape(x)[2]
        x = tf.image.resize_images(x, (h*2, w*2), method=tf.image.ResizeMethod.BILINEAR)
        x = slim.convolution2d(x, channel, [3, 3], activation_fn=None)
        x = slim.convolution2d(x, channel, [3, 3], activation_fn=None)
        
        x = tf.contrib.layers.instance_norm(x)
        x = tf.nn.relu(x)
        
        '''
        x = slim.convolution2d(x, channel*4, [3, 3], activation_fn=None)
        x = layers.pixel_shuffle(x, 2, channel)
        x = slim.convolution2d(x, channel, [3, 3], activation_fn=None)
        
        x = slim.conv2d_transpose(x, channel, [3, 3], stride=2, activation_fn=None)
        x = slim.conv2d_transpose(x, channel, [3, 3], activation_fn=None)
        '''
        h, w = tf.shape(x)[1], tf.shape(x)[2]
        x = tf.image.resize_images(x, (h*2, w*2), method=tf.image.ResizeMethod.BILINEAR)
        x = slim.convolution2d(x, channel, [3, 3], activation_fn=None)
        x = slim.convolution2d(x, channel, [3, 3], activation_fn=None)
        
        x = tf.contrib.layers.instance_norm(x)
        x = tf.nn.relu(x)
        
        x = slim.convolution2d(x, 3, [7, 7], activation_fn=None)
        #x = tf.nn.tanh(x)
        x = tf.clip_by_value(x, -1, 1)
        
        return x
    
    
def generator_bn(inputs, channel=32, num_blocks=4, is_training=False, name='generator'):
    with tf.variable_scope(name):
        
        x = slim.convolution2d(inputs, channel, [7, 7], activation_fn=None)
        x = slim.batch_norm(x, is_training=is_training, center=True, scale=True)
        x = tf.nn.relu(x)
        
        x = slim.convolution2d(x, channel*2, [3, 3], stride=2, activation_fn=None)
        x = slim.convolution2d(x, channel*2, [3, 3], activation_fn=None)
        x = slim.batch_norm(x, is_training=is_training, center=True, scale=True)
        x = tf.nn.relu(x)
       
        x = slim.convolution2d(x, channel*4, [3, 3], stride=2, activation_fn=None)
        x = slim.convolution2d(x, channel*4, [3, 3], activation_fn=None)
        x = slim.batch_norm(x, is_training=is_training, center=True, scale=True)
        x = tf.nn.relu(x)
        
        
        for idx in range(num_blocks):
            x = layers.resblock_bn(x, channel*4, is_training, name='block_{}'.format(idx))
            
        '''
        x = slim.conv2d_transpose(x, channel*2, [3, 3], stride=2, activation_fn=None)
        x = slim.conv2d_transpose(x, channel*2, [3, 3], activation_fn=None)
        '''
        h, w = tf.shape(x)[1], tf.shape(x)[2]
        x = tf.image.resize_images(x, (h*2, w*2), method=tf.image.ResizeMethod.BILINEAR)
        x = slim.convolution2d(x, channel*2, [3, 3], activation_fn=None)
        x = slim.convolution2d(x, channel*2, [3, 3], activation_fn=None)
        
        x = slim.batch_norm(x, is_training=is_training, center=True, scale=True)
        x = tf.nn.relu(x)
        
        '''
        x = slim.conv2d_transpose(x, channel, [3, 3], stride=2, activation_fn=None)
        x = slim.conv2d_transpose(x, channel, [3, 3], activation_fn=None)
        '''
        h, w = tf.shape(x)[1], tf.shape(x)[2]
        x = tf.image.resize_images(x, (h*2, w*2), method=tf.image.ResizeMethod.BILINEAR)
        x = slim.convolution2d(x, channel, [3, 3], activation_fn=None)
        x = slim.convolution2d(x, channel, [3, 3], activation_fn=None)
        
        x = slim.batch_norm(x, is_training=is_training, center=True, scale=True)
        x = tf.nn.relu(x)
        
        x = slim.convolution2d(x, 3, [7, 7], activation_fn=None)
        #x = tf.nn.tanh(x)
        x = tf.clip_by_value(x, -1, 1)
        
        return x
            
            
    
def discriminator(inputs, name='discriminator', reuse=False):
    with tf.variable_scope(name, reuse=reuse):
        
        h, w, c = inputs.get_shape().as_list()[1:]
        batch_size = tf.shape(inputs)[0]
        patch = tf.random_crop(inputs, [batch_size, h//4, w//4, c])
        
        x = slim.convolution2d(patch, 32, [3, 3], activation_fn=None)
        x = tf.nn.leaky_relu(x)
        
        x = slim.convolution2d(x, 64, [3, 3], stride=2, activation_fn=None)
        x = tf.nn.leaky_relu(x)
        
        x = slim.convolution2d(x, 128, [3, 3], activation_fn=None)
        x = tf.nn.leaky_relu(tf.contrib.layers.instance_norm(x))
        
        x = slim.convolution2d(x, 128, [3, 3], stride=2, activation_fn=None)
        x = tf.nn.leaky_relu(x)
        
        x = slim.convolution2d(x, 256, [3, 3], activation_fn=None)
        x = tf.nn.leaky_relu(tf.contrib.layers.instance_norm(x))
        
        x = slim.convolution2d(x, 256, [3, 3], activation_fn=None)
        x = tf.nn.leaky_relu(tf.contrib.layers.instance_norm(x))
        
        x = slim.convolution2d(x, 1, [3, 3], activation_fn=None)
        x = tf.nn.sigmoid(x)
                
        return x


def discriminator_bn(inputs, is_training=False, reuse=False, name='discriminator'):
    with tf.variable_scope(name, reuse=reuse):
        
        h, w, c = inputs.get_shape().as_list()[1:]
        batch_size = tf.shape(inputs)[0]
        patch = tf.random_crop(inputs, [batch_size, h//4, w//4, c])
        
        x = slim.convolution2d(patch, 32, [3, 3], stride=2, activation_fn=None)
        x = slim.batch_norm(x, is_training=is_training, center=True, scale=True)
        x = tf.nn.leaky_relu(x)
        
        x = slim.convolution2d(patch, 32, [3, 3], activation_fn=None)
        x = slim.batch_norm(x, is_training=is_training, center=True, scale=True)
        x = tf.nn.leaky_relu(x)
        
        x = slim.convolution2d(x, 64, [3, 3], stride=2, activation_fn=None)
        x = slim.batch_norm(x, is_training=is_training, center=True, scale=True)
        x = tf.nn.leaky_relu(x)
        
        x = slim.convolution2d(x, 64, [3, 3], activation_fn=None)
        x = slim.batch_norm(x, is_training=is_training, center=True, scale=True)
        x = tf.nn.leaky_relu(x)
        
        x = slim.convolution2d(x, 128, [3, 3], stride=2, activation_fn=None)
        x = slim.batch_norm(x, is_training=is_training, center=True, scale=True)
        x = tf.nn.leaky_relu(x)
        
        x = slim.convolution2d(x, 128, [3, 3], activation_fn=None)
        x = slim.batch_norm(x, is_training=is_training, center=True, scale=True)
        x = tf.nn.leaky_relu(x)
        
        x = tf.reduce_mean(x, axis=[1, 2])
        x = slim.fully_connected(x, 1, activation_fn=None)
        x = tf.nn.sigmoid(x)
                
        return x
            
            
def disc_wgan(inputs, name='disc_wgan', reuse=False):
    
    with tf.variable_scope(name, reuse=reuse):
        
        h, w, c = inputs.get_shape().as_list()[1:]
        batch_size = tf.shape(inputs)[0]
        patch = tf.random_crop(inputs, [batch_size, h//4, w//4, c])
        
        x = slim.convolution2d(patch, 32, [3, 3], activation_fn=None)
        x = tf.nn.leaky_relu(tf.contrib.layers.layer_norm(x))
        
        x = slim.convolution2d(x, 32, [3, 3], stride=2, activation_fn=None)
        x = tf.nn.leaky_relu(tf.contrib.layers.layer_norm(x))
 
        x = slim.convolution2d(x, 64, [3, 3], activation_fn=None)
        x = tf.nn.leaky_relu(tf.contrib.layers.layer_norm(x))
        
        x = slim.convolution2d(x, 64, [3, 3], stride=2, activation_fn=None)
        x = tf.nn.leaky_relu(tf.contrib.layers.layer_norm(x))
        
        x = slim.convolution2d(x, 128, [3, 3], activation_fn=None)
        x = tf.nn.leaky_relu(tf.contrib.layers.layer_norm(x))
        
        x = slim.convolution2d(x, 128, [3, 3], stride=2, activation_fn=None)
        x = tf.nn.leaky_relu(tf.contrib.layers.layer_norm(x))
        
        '''
        x = slim.convolution2d(x, 256, [3, 3], activation_fn=None)
        x = tf.nn.leaky_relu(tf.contrib.layers.layer_norm(x))
        
        x = slim.convolution2d(x, 256, [3, 3], stride=2, activation_fn=None)
        x = tf.nn.leaky_relu(tf.contrib.layers.layer_norm(x))
        '''
        x = tf.reduce_mean(x, axis=[1, 2])
        x = slim.fully_connected(x, 1, activation_fn=None)
        
        return x
    
         
            
if __name__ == '__main__':
    

    inputs = tf.placeholder(tf.float32, [1, 256, 256, 3])
    outputs = generator_bn(inputs)
    #print(outputs.get_shape().as_list())

    
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    
    converter = tf.contrib.lite.TFLiteConverter.from_session(
                            sess, [inputs], [outputs])
    '''
    converter.default_ranges_stats=(0, 6)
    converter.inference_type = tf.lite.constants.QUANTIZED_UINT8
    input_arrays = converter.get_input_arrays()
    converter.quantized_input_stats = {input_arrays[0] : (0., 1.)}  # mean, std_dev
    '''
    tflite_model = converter.convert()
    open("test.tflite", "wb").write(tflite_model)
    
   