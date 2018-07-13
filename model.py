import tensorflow as tf
import tensorflow.contrib.slim as slim
import utils

batch_norm = tf.layers.batch_normalization 



def res_block(input_tensor, channel, is_train=False):
    short_cut = input_tensor
    res_conv1 = slim.conv2d(input_tensor, channel, [3, 3], activation_fn=None)
    res_norm1 = batch_norm(res_conv1, training=is_train)
    res_relu1 = tf.nn.relu(res_norm1)
    res_conv2 = slim.conv2d(res_relu1, channel, [3, 3], activation_fn=None)
    res_norm2 = batch_norm(res_conv2, training=is_train)
    return res_norm2 + short_cut


def generator(input_tensor, name='generator', reuse=False, is_train=False):
    
    with tf.variable_scope(name, reuse=reuse):
        
        conv1 = slim.conv2d(input_tensor, 32, [7, 7], activation_fn=None)
        norm1 = batch_norm(conv1, training=is_train)
        relu1 = tf.nn.relu(norm1)

        conv2_1 = slim.conv2d(relu1, 64, [3, 3], stride=2,  activation_fn=None)
        conv2_2 = slim.conv2d(conv2_1, 64, [3, 3],  activation_fn=None)
        norm2 = batch_norm(conv2_2, training=is_train)
        relu2 = tf.nn.relu(norm2)

        conv3_1 = slim.conv2d(relu2, 128, [3, 3], stride=2,  activation_fn=None)
        conv3_2 = slim.conv2d(conv3_1, 128, [3, 3],  activation_fn=None)
        norm3 = batch_norm(conv3_2, training=is_train)
        relu3 = tf.nn.relu(norm3)
        
        x = relu3
        for _ in range(4):
            x = res_block(x, 128, is_train=is_train)

        conv_up1_1 = slim.conv2d_transpose(x, 64, [3, 3], stride=2,  activation_fn=None)
        conv_up1_2 = slim.conv2d_transpose(conv_up1_1, 64, [3, 3],  activation_fn=None)
        norm4 = batch_norm(conv_up1_2, training=is_train)
        relu4 = tf.nn.relu(norm4)

        conv_up2_1 = slim.conv2d_transpose(relu4, 32, [3, 3], stride=2,  activation_fn=None)
        conv_up2_2 = slim.conv2d_transpose(conv_up2_1, 32, [3, 3],  activation_fn=None)
        norm5 = batch_norm(conv_up2_2, training=is_train)
        relu5 = tf.nn.relu(norm5)

        conv_out = slim.conv2d(relu5, 3, [7, 7], activation_fn=None)

        return conv_out


def discriminator(input_tensor, patch_size, name='discriminator', reuse=False, is_train=False):
    #input size 36*36 
    with tf.variable_scope(name, reuse=reuse):
            
        conv1 = slim.conv2d(input_tensor, 32, [3, 3], activation_fn=None)
        norm1 = tf.contrib.layers.layer_norm(conv1)
        relu1 = utils.leaky_relu(norm1)
        
        conv2 = slim.conv2d(relu1, 32, [3, 3], stride=2, activation_fn=None)
        norm2 = tf.contrib.layers.layer_norm(conv2)
        relu2 = utils.leaky_relu(norm2)
        
        conv3 = slim.conv2d(relu2, 64, [3, 3], activation_fn=None)
        norm3 = tf.contrib.layers.layer_norm(conv3)
        relu3 = utils.leaky_relu(norm3)
        
        conv4 = slim.conv2d(relu3, 64, [3, 3], stride=2, activation_fn=None)
        norm4 = tf.contrib.layers.layer_norm(conv4)
        relu4 = utils.leaky_relu(norm4)
      
        conv5 = slim.conv2d(relu4, 128, [3, 3], activation_fn=None)
        norm5 = tf.contrib.layers.layer_norm(conv5)
        relu5 = utils.leaky_relu(norm5)
        
        conv6 = slim.conv2d(relu5, 128, [3, 3], stride=2, activation_fn=None)
        norm6 = tf.contrib.layers.layer_norm(conv6)
        relu6 = utils.leaky_relu(norm6)

        conv7 = slim.conv2d(relu6, 256, [3, 3], stride=2, activation_fn=None)
        norm7 = tf.contrib.layers.layer_norm(conv7)
        relu7 = utils.leaky_relu(norm7)

        conv8 = slim.conv2d(relu7, 256, [3, 3], stride=2, activation_fn=None)
        norm8 = tf.contrib.layers.layer_norm(conv8)
        relu8 = utils.leaky_relu(norm8)
        
        conv_out = slim.conv2d(relu8, 1, [1, 1], activation_fn=None)
        avg_pool = tf.reduce_mean(conv_out, axis=[1, 2])
        #sprint(avg_pool.get_shape())
        
        return avg_pool


def patch_discriminator(input_tensor, patch_size, name='discriminator', reuse=False):
    #input size 32*32
    with tf.variable_scope(name, reuse=reuse):
            
        batch_size = tf.shape(input_tensor)[0]
        patch = tf.random_crop(input_tensor, [batch_size, patch_size, patch_size, 3])
        
        conv1 = slim.conv2d(patch, 32, [3, 3], activation_fn=None)
        norm1 = tf.contrib.layers.layer_norm(conv1)
        relu1 = utils.leaky_relu(norm1)
        
        conv2 = slim.conv2d(relu1, 32, [3, 3], stride=2, activation_fn=None)
        norm2 = tf.contrib.layers.layer_norm(conv2)
        relu2 = utils.leaky_relu(norm2)
        
        conv3 = slim.conv2d(relu2, 64, [3, 3], activation_fn=None)
        norm3 = tf.contrib.layers.layer_norm(conv3)
        relu3 = utils.leaky_relu(norm3)
        
        conv4 = slim.conv2d(relu3, 64, [3, 3], stride=2, activation_fn=None)
        norm4 = tf.contrib.layers.layer_norm(conv4)
        relu4 = utils.leaky_relu(norm4)
        
        conv5 = slim.conv2d(relu4, 128, [3, 3], activation_fn=None)
        norm5 = tf.contrib.layers.layer_norm(conv5)
        relu5 = utils.leaky_relu(norm5)
        
        conv6 = slim.conv2d(relu5, 128, [3, 3], stride=2, activation_fn=None)
        norm6 = tf.contrib.layers.layer_norm(conv6)
        relu6 = utils.leaky_relu(norm6)
        
        conv_out = slim.conv2d(relu6, 1, [1, 1], activation_fn=None)
        avg_pool = tf.reduce_mean(conv_out, axis=[1, 2])
        #sprint(avg_pool.get_shape())
        
        return avg_pool

