import os
import cv2
import numpy as np
import scipy.stats as st
import tensorflow as tf


VGG_MEAN = [103.939, 116.779, 123.68]


class Vgg19:
    
    def __init__(self, vgg19_npy_path=None):
        
        self.data_dict = np.load(vgg19_npy_path, encoding='latin1', allow_pickle=True).item()
        print('Finished loading vgg19.npy')


    def build_relu5_4(self, rgb, include_fc=False):
        
        rgb_scaled = (rgb+1) * 127.5

        blue, green, red = tf.split(axis=3, num_or_size_splits=3, value=rgb_scaled)
        bgr = tf.concat(axis=3, values=[blue - VGG_MEAN[0],
                        green - VGG_MEAN[1], red - VGG_MEAN[2]])

        self.conv1_1 = self.conv_layer(bgr, "conv1_1")
        self.relu1_1 = tf.nn.relu(self.conv1_1)
        self.conv1_2 = self.conv_layer(self.relu1_1, "conv1_2")
        self.relu1_2 = tf.nn.relu(self.conv1_2)
        self.pool1 = self.max_pool(self.relu1_2, 'pool1')

        self.conv2_1 = self.conv_layer(self.pool1, "conv2_1")
        self.relu2_1 = tf.nn.relu(self.conv2_1)
        self.conv2_2 = self.conv_layer(self.relu2_1, "conv2_2")
        self.relu2_2 = tf.nn.relu(self.conv2_2)
        self.pool2 = self.max_pool(self.relu2_2, 'pool2')

        self.conv3_1 = self.conv_layer(self.pool2, "conv3_1")
        self.relu3_1 = tf.nn.relu(self.conv3_1)
        self.conv3_2 = self.conv_layer(self.relu3_1, "conv3_2")
        self.relu3_2 = tf.nn.relu(self.conv3_2)
        self.conv3_3 = self.conv_layer(self.relu3_2, "conv3_3")
        self.relu3_3 = tf.nn.relu(self.conv3_3)
        self.conv3_4 = self.conv_layer(self.relu3_3, "conv3_4")
        self.relu3_4 = tf.nn.relu(self.conv3_4)
        self.pool3 = self.max_pool(self.relu3_4, 'pool3')

        self.conv4_1 = self.conv_layer(self.pool3, "conv4_1")
        self.relu4_1 = tf.nn.relu(self.conv4_1)
        self.conv4_2 = self.conv_layer(self.relu4_1, "conv4_2")
        self.relu4_2 = tf.nn.relu(self.conv4_2)
        self.conv4_3 = self.conv_layer(self.relu4_2, "conv4_3")
        self.relu4_3 = tf.nn.relu(self.conv4_3)
        self.conv4_4 = self.conv_layer(self.relu4_3, "conv4_4")
        self.relu4_4 = tf.nn.relu(self.conv4_4)
        self.pool4 = self.max_pool(self.relu4_4, 'pool4')

        self.conv5_1 = self.conv_layer(self.pool4, "conv5_1")
        self.relu5_1 = tf.nn.relu(self.conv5_1)
        self.conv5_2 = self.conv_layer(self.relu5_1, "conv5_2")
        self.relu5_2 = tf.nn.relu(self.conv5_2)
        self.conv5_3 = self.conv_layer(self.relu5_2, "conv5_3")
        self.relu5_3 = tf.nn.relu(self.conv5_3)
        self.conv5_4 = self.conv_layer(self.relu5_3, "conv5_4")
        self.relu5_4 = tf.nn.relu(self.conv5_4)
        self.pool5 = self.max_pool(self.relu5_4, 'pool5')
        
        return self.conv5_4
    
    
    def build_conv4_4(self, rgb, include_fc=False):
        
        rgb_scaled = (rgb+1) * 127.5
      
        blue, green, red = tf.split(axis=3, num_or_size_splits=3, value=rgb_scaled)
        bgr = tf.concat(axis=3, values=[blue - VGG_MEAN[0],
                        green - VGG_MEAN[1], red - VGG_MEAN[2]])

        self.conv1_1 = self.conv_layer(bgr, "conv1_1")
        self.relu1_1 = tf.nn.relu(self.conv1_1)
        self.conv1_2 = self.conv_layer(self.relu1_1, "conv1_2")
        self.relu1_2 = tf.nn.relu(self.conv1_2)
        self.pool1 = self.max_pool(self.relu1_2, 'pool1')

        self.conv2_1 = self.conv_layer(self.pool1, "conv2_1")
        self.relu2_1 = tf.nn.relu(self.conv2_1)
        self.conv2_2 = self.conv_layer(self.relu2_1, "conv2_2")
        self.relu2_2 = tf.nn.relu(self.conv2_2)
        self.pool2 = self.max_pool(self.relu2_2, 'pool2')

        self.conv3_1 = self.conv_layer(self.pool2, "conv3_1")
        self.relu3_1 = tf.nn.relu(self.conv3_1)
        self.conv3_2 = self.conv_layer(self.relu3_1, "conv3_2")
        self.relu3_2 = tf.nn.relu(self.conv3_2)
        self.conv3_3 = self.conv_layer(self.relu3_2, "conv3_3")
        self.relu3_3 = tf.nn.relu(self.conv3_3)
        self.conv3_4 = self.conv_layer(self.relu3_3, "conv3_4")
        self.relu3_4 = tf.nn.relu(self.conv3_4)
        self.pool3 = self.max_pool(self.relu3_4, 'pool3')

        self.conv4_1 = self.conv_layer(self.pool3, "conv4_1")
        self.relu4_1 = tf.nn.relu(self.conv4_1)
        self.conv4_2 = self.conv_layer(self.relu4_1, "conv4_2")
        self.relu4_2 = tf.nn.relu(self.conv4_2)
        self.conv4_3 = self.conv_layer(self.relu4_2, "conv4_3")
        self.relu4_3 = tf.nn.relu(self.conv4_3)
        self.conv4_4 = self.conv_layer(self.relu4_3, "conv4_4")
        
        return self.conv4_4
    
    
    def build_conv2_2(self, rgb, include_fc=False):
        
        rgb_scaled = (rgb+1) * 127.5
      
        blue, green, red = tf.split(axis=3, num_or_size_splits=3, value=rgb_scaled)
        bgr = tf.concat(axis=3, values=[blue - VGG_MEAN[0],
                        green - VGG_MEAN[1], red - VGG_MEAN[2]])

        self.conv1_1 = self.conv_layer(bgr, "conv1_1")
        self.relu1_1 = tf.nn.relu(self.conv1_1)
        self.conv1_2 = self.conv_layer(self.relu1_1, "conv1_2")
        self.relu1_2 = tf.nn.relu(self.conv1_2)
        self.pool1 = self.max_pool(self.relu1_2, 'pool1')

        self.conv2_1 = self.conv_layer(self.pool1, "conv2_1")
        self.relu2_1 = tf.nn.relu(self.conv2_1)
        self.conv2_2 = self.conv_layer(self.relu2_1, "conv2_2")
        #self.relu2_2 = tf.nn.relu(self.conv2_2)
        
        return self.conv2_2
            
    
    def max_pool(self, bottom, name):
        return tf.nn.max_pool(bottom, ksize=[1, 2, 2, 1], 
                    strides=[1, 2, 2, 1], padding='SAME', name=name)

    def conv_layer(self, bottom, name):
        with tf.variable_scope(name):
            filt = self.get_conv_filter(name)

            conv = tf.nn.conv2d(bottom, filt, [1, 1, 1, 1], padding='SAME')

            conv_biases = self.get_bias(name)
            bias = tf.nn.bias_add(conv, conv_biases)

            #relu = tf.nn.relu(bias)
            return bias

    def fc_layer(self, bottom, name):
        with tf.variable_scope(name):
            shape = bottom.get_shape().as_list()
            dim = 1
            for d in shape[1:]:
                dim *= d
            x = tf.reshape(bottom, [-1, dim])

            weights = self.get_fc_weight(name)
            biases = self.get_bias(name)

            # Fully connected layer. Note that the '+' operation automatically
            # broadcasts the biases.
            fc = tf.nn.bias_add(tf.matmul(x, weights), biases)

            return fc

    def get_conv_filter(self, name):
        return tf.constant(self.data_dict[name][0], name="filter")

    def get_bias(self, name):
        return tf.constant(self.data_dict[name][1], name="biases")

    def get_fc_weight(self, name):
        return tf.constant(self.data_dict[name][0], name="weights")


def vggloss_5_4(image_a, image_b):
    vgg_model = Vgg19('vgg19_no_fc.npy')
    vgg_a = vgg_model.build_relu5_4(image_a)
    vgg_b = vgg_model.build_relu5_4(image_b)
    #VGG_loss = tf.losses.absolute_difference(vgg_a, vgg_b)
    VGG_loss = tf.nn.l2_loss(vgg_a - vgg_b)
    h, w, c= vgg_a.get_shape().as_list()[1:]
    VGG_loss = tf.reduce_mean(VGG_loss)/(h*w*c)
    return VGG_loss

def vggloss_4_4(image_a, image_b):
    vgg_model = Vgg19('vgg19_no_fc.npy')
    vgg_a = vgg_model.build_conv4_4(image_a)
    vgg_b = vgg_model.build_conv4_4(image_b)
    VGG_loss = tf.losses.absolute_difference(vgg_a, vgg_b)
    #VGG_loss = tf.nn.l2_loss(vgg_a - vgg_b)
    h, w, c= vgg_a.get_shape().as_list()[1:]
    VGG_loss = tf.reduce_mean(VGG_loss)/(h*w*c)
    return VGG_loss



def vggloss_2_2(image_a, image_b):
    vgg_model = Vgg19('vgg19_no_fc.npy')
    vgg_a = vgg_model.build_conv2_2(image_a)
    vgg_b = vgg_model.build_conv2_2(image_b)
    VGG_loss = tf.losses.absolute_difference(vgg_a, vgg_b)
    #VGG_loss = tf.nn.l2_loss(vgg_a - vgg_b)
    h, w, c= vgg_a.get_shape().as_list()[1:]
    VGG_loss = tf.reduce_mean(VGG_loss)/(h*w*c)
    return VGG_loss



def gauss_kernel(k_size=21, nsig=3, channels=1):
    interval = (2*nsig+1.)/(k_size)
    x = np.linspace(-nsig-interval/2., nsig+interval/2., k_size+1)
    kern1d = np.diff(st.norm.cdf(x))
    kernel_raw = np.sqrt(np.outer(kern1d, kern1d))
    kernel = kernel_raw/kernel_raw.sum()
    out_filter = np.array(kernel, dtype = np.float32)
    out_filter = out_filter.reshape((k_size, k_size, 1, 1))
    out_filter = np.repeat(out_filter, channels, axis = 2)
    
    return out_filter


def blur(image):
    kernel_var = gauss_kernel(21, 3, 3)
    return tf.nn.depthwise_conv2d(image, kernel_var, 
                                  [1, 1, 1, 1], padding='SAME')


def wgan_loss(discriminator, real, fake, name='discriminator', lambda_=10):
    real_logits = discriminator(real, name=name, reuse=False)
    fake_logits = discriminator(fake, name=name, reuse=True)

    d_loss_real = - tf.reduce_mean(real_logits)
    d_loss_fake = tf.reduce_mean(fake_logits)

    d_loss = d_loss_real + d_loss_fake
    g_loss = - d_loss_fake

    """ Gradient Penalty """
    # This is borrowed from https://github.com/kodalinaveen3/DRAGAN/blob/master/DRAGAN.ipynb
    alpha = tf.random_uniform([tf.shape(real)[0], 1, 1, 1], minval=0.,maxval=1.)
    differences = fake - real # This is different from MAGAN
    interpolates = real + (alpha * differences)
    inter_logit = discriminator(interpolates, name=name, reuse=True)
    gradients = tf.gradients(inter_logit, [interpolates])[0]
    slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
    gradient_penalty = tf.reduce_mean((slopes - 1.) ** 2)
    d_loss += lambda_ * gradient_penalty
    
    return d_loss, g_loss



def load_image_list(data_dir):
    name_list = list()
    for name in os.listdir(data_dir):
        name_list.append(os.path.join(data_dir, name))
    return name_list



def next_batch(filename_list, batch_size, crop_size):
    idx = np.arange(0 , len(filename_list))
    np.random.shuffle(idx)
    idx = idx[:batch_size]
    batch_data = []
    for i in range(batch_size):
        try:
            image = cv2.imread(filename_list[idx[i]])
            img_h, img_w = np.shape(image)[: 2]
            offset_h = np.random.randint(0, img_h - crop_size)
            offset_w = np.random.randint(0, img_w - crop_size)
            image_crop = image[offset_h: offset_h + crop_size, 
                                    offset_w: offset_w + crop_size]
            #image_crop = image_crop[:, :, ::-1]
            batch_data.append(image_crop/127.5-1)
        except:
            print(filename_list[idx[i]])
            print(np.shape(image))
    return np.asarray(batch_data)


def write_batch_image(image, save_dir, name, n):
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
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
    

if __name__ == '__main__':
    from tqdm import tqdm

    cartoon_dir = 'C:\\Users\\Razer\\Downloads\\dataset\\cartoon_jpg'
    photo_dir = 'C:\\Users\\Razer\\Downloads\\dataset\\celeba_jpg'
    photo_list = load_image_list(photo_dir)
    cartoon_list = load_image_list(cartoon_dir)

    for name in tqdm(photo_list):
        image = cv2.imread(name)
        h, w, c = np.shape(image)
        if h > w:
            h, w = np.round(100*h/w).astype(np.uint8), 100
            
        elif h < w:
            h, w = 100, np.round(100*w/h).astype(np.uint8)
        image = cv2.resize(image, (w, h),
                           interpolation=cv2.INTER_CUBIC)
        cv2.imwrite(name, image)
            
    '''
    for name in tqdm(cartoon_list):
        try:
            h, w, c = np.shape(image)
        except:
            os.remove(name)
    '''
    
    
    
    