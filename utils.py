import tensorflow as tf
import numpy as np
import cv2
import os
from vgg19 import Vgg19


def leaky_relu(x, leak=0.2):
    return tf.maximum(x, leak*x)


def print_image(image, save_dir, name):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    fused_dir = os.path.join(save_dir, 'fused_image.jpg')
    fused_image = [0] * 8
    for i in range(8):
        fused_image[i] = []
        for j in range(8):
            k = i * 8 + j
            image[k] = (image[k] + 1) * 127.5
            fused_image[i].append(image[k])
            img_dir = os.path.join(save_dir, name+str(k)+'.jpg')
            cv2.imwrite(img_dir, image[k])
        fused_image[i] = np.hstack(fused_image[i])
        #fused_image[i] = np.concatenate(fused_image[i], axis = 1)
    fused_image = np.vstack(fused_image)
    cv2.imwrite(fused_dir, fused_image)
    

def print_fused_image(image, save_dir, name, n):
    '''
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    '''
    fused_dir = os.path.join(save_dir, name)
    fused_image = [0] * n
    for i in range(n):
        fused_image[i] = []
        for j in range(n):
            k = i * n + j
            image[k] = (image[k] + 1) * 127.5
            fused_image[i].append(image[k])
        fused_image[i] = np.hstack(fused_image[i])
    fused_image = np.vstack(fused_image)
    cv2.imwrite(fused_dir, fused_image)


def get_filename_list(load_dir):

    filename_list = []
    for name in os.listdir(load_dir):
        file_name = os.path.join(load_dir, name)
        filename_list.append(file_name)

    return filename_list



def next_batch(batch_size, crop_size, filename_list):
    idx = np.arange(0 , len(filename_list))
    np.random.shuffle(idx)
    idx = idx[:batch_size]
    batch_data = []
    for i in range(batch_size):
        image = cv2.imread(filename_list[idx[i]])
        img_h, img_w = np.shape(image)[: 2]
        offset_h = np.random.randint(0, img_h - crop_size)
        offset_w = np.random.randint(0, img_w - crop_size)
        image_crop = image[offset_h: offset_h + crop_size, 
                                offset_w: offset_w + crop_size]
        batch_data.append(image_crop/127.5-1)
    
    return np.asarray(batch_data)


def next_blur_batch(batch_size, crop_size, filename_list):
    idx = np.arange(0 , len(filename_list))
    np.random.shuffle(idx)
    idx = idx[:batch_size]
    batch, blur_batch = [], []
    for i in range(batch_size):
        image = cv2.imread(filename_list[idx[i]])
        img_h, img_w = np.shape(image)[: 2]
        offset_h = np.random.randint(0, img_h - crop_size)
        offset_w = np.random.randint(0, img_w - crop_size)
        image_crop = image[offset_h: offset_h + crop_size, 
                                offset_w: offset_w + crop_size]
        batch.append(image_crop/127.5-1)
        image_blur = cv2.GaussianBlur(image_crop, (5, 5), 0)
        blur_batch.append(image_blur/127.5-1)
    
    return np.asarray(batch), np.asarray(blur_batch)


def vgg_loss(image_a, image_b):
    vgg_a, vgg_b = Vgg19('vgg19.npy'), Vgg19('vgg19.npy')
    vgg_a.build(image_a)
    vgg_b.build(image_b)
    VGG_loss = tf.reduce_mean(tf.losses.absolute_difference(vgg_a.conv4_4, vgg_b.conv4_4))
    h = tf.cast(tf.shape(vgg_a.conv4_4)[1], tf.float32)
    w = tf.cast(tf.shape(vgg_a.conv4_4)[2], tf.float32)
    c = tf.cast(tf.shape(vgg_a.conv4_4)[3], tf.float32)
    VGG_loss = VGG_loss/(h*w*c)
    return VGG_loss


def l2_norm(v, eps=1e-12):
    return v / (tf.reduce_sum(v ** 2) ** 0.5 + eps)

    
#celeba = load_celeba('D:/celeba_dataset')
def spectral_norm(w, iteration=1):
    w_shape = w.shape.as_list()
    w = tf.reshape(w, [-1, w_shape[-1]])

    u = tf.get_variable("u", [1, w_shape[-1]], initializer=
                        tf.truncated_normal_initializer(), trainable=False)

    u_hat = u
    v_hat = None
    for i in range(iteration):
        v_ = tf.matmul(u_hat, tf.transpose(w))
        v_hat = l2_norm(v_)

        u_ = tf.matmul(v_hat, w)
        u_hat = l2_norm(u_)

    sigma = tf.matmul(tf.matmul(v_hat, w), tf.transpose(u_hat))
    w_norm = w / sigma

    with tf.control_dependencies([u.assign(u_hat)]):
        w_norm = tf.reshape(w_norm, w_shape)
    
    return w_norm



def conv_sn(x, channels, k_size, stride=1, name='conv2d'):
    with tf.variable_scope(name):
        w = tf.get_variable("kernel", shape=[k_size, k_size, x.get_shape()[-1], channels],
                            initializer=tf.contrib.layers.xavier_initializer())
        b = tf.get_variable("bias", [channels], initializer=tf.constant_initializer(0.0))

        output = tf.nn.conv2d(input=x, filter=spectral_norm(w), 
                        strides=[1, stride, stride, 1], padding='SAME') + b
        return output