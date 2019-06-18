import os
import tensorflow as tf
import numpy as np
import argparse
from network import generator, disc_wgan
from dataset import cartoon_loader
from utils import vggloss_5_4, blur, wgan_loss
from tqdm import tqdm


os.environ["CUDA_VISIBLE_DEVICES"]="0"


def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--patch_size", default = 96, type = int)
    parser.add_argument("--batch_size", default = 32, type = int)     
    parser.add_argument("--total_epoch", default = 50, type = int)
    parser.add_argument("--learning_rate", default = 1e-4, type = float)
    parser.add_argument("--gpu_fraction", default = 0.75, type = float)
    parser.add_argument("--save_model_dir", default = 'saved_models')
    parser.add_argument("--train_log_dir", default = 'train_log')
    parser.add_argument("--mode", default = 'train')
    
    args = parser.parse_args()
    return args

def train(args):
    

    input_image = tf.placeholder(tf.float32, [None, args.patch_size, args.patch_size, 3])
    ref_image = tf.placeholder(tf.float32, [None, args.patch_size, args.patch_size, 3])
    learning_rate = tf.placeholder(tf.float32)
    
    result_image = generator(input_image, 'gen_forward', is_training=True)
    reverse_image = generator(result_image, 'gen_backward', is_training=True)
    
    result_blur = blur(result_image)
    reference_blur = blur(ref_image)
    result_gray = tf.image.rgb_to_grayscale(result_image)
    reference_gray = tf.image.rgb_to_grayscale(ref_image)
    
    blur_d_loss, blur_g_loss = wgan_loss(disc_wgan, real=reference_blur, 
                                         fake=result_blur, name='disc_blur')
    gray_d_loss, gray_g_loss = wgan_loss(disc_wgan, real=reference_gray, 
                                         fake=result_gray, name='disc_gray')
    
    '''
    blur_fake = discriminator(result_blur, 'disc_blur', is_training=True, reuse=False)
    blur_real = discriminator(reference_blur, 'disc_blur', is_training=True, reuse=True)
    
    gray_fake = discriminator(result_gray, 'disc_gray', is_training=True, reuse=False)
    gray_real = discriminator(reference_gray, 'disc_gray', is_training=True, reuse=True)
    
    blur_real_loss = -tf.reduce_mean(tf.log(tf.nn.sigmoid(blur_real) + 1e-8))
    blur_fake_loss = tf.reduce_mean(tf.log(1. - tf.nn.sigmoid(blur_fake) + 1e-8))
    blur_d_loss = blur_real_loss + blur_fake_loss
    blur_g_loss = -tf.reduce_mean(tf.log(tf.nn.sigmoid(blur_fake) + 1e-8))
    
    
    gray_real_loss = -tf.reduce_mean(tf.log(tf.nn.sigmoid(gray_real) + 1e-8))
    gray_fake_loss = tf.reduce_mean(tf.log(1. - tf.nn.sigmoid(gray_fake)) + 1e-8)
    gray_d_loss = gray_real_loss + gray_fake_loss
    gray_g_loss = -tf.reduce_mean(tf.log(tf.nn.sigmoid(gray_fake) + 1e-8))
    '''
    
    tv_h = tf.reduce_mean((result_image[:, 1:, :, :] - 
                           result_image[:, :args.patch_size - 1, :, :])**2)
    tv_w = tf.reduce_mean((result_image[:, :, 1:, :] - 
                           result_image[:, :, :args.patch_size - 1, :])**2)
    tv_loss = (tv_h + tv_w)/(3*args.patch_size**2)
    #tv_loss = tv_h + tv_w
    
    content_loss = vggloss_5_4(result_image, reverse_image)
    
    g_loss_total = 10*tv_loss + content_loss + 5e-3*(gray_g_loss + blur_g_loss)
    d_loss_total = gray_d_loss + blur_d_loss
    
    all_vars = tf.trainable_variables()
    gen_vars = [var for var in all_vars if 'gen' in var.name]
    disc_vars = [var for var in all_vars if 'disc' in var.name]

    tf.summary.scalar('content_loss', content_loss)
    tf.summary.scalar('tv_loss', tv_loss)
    tf.summary.scalar('texture_loss', gray_g_loss)
    tf.summary.scalar('color_loss', blur_g_loss)
    tf.summary.scalar('generator_loss', g_loss_total)
    tf.summary.scalar('discriminator_loss', d_loss_total)
    
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        g_optim = tf.train.AdamOptimizer(learning_rate)\
                            .minimize(g_loss_total, var_list=gen_vars)
        d_optim = tf.train.AdamOptimizer(learning_rate)\
                            .minimize(d_loss_total, var_list=disc_vars)
    '''
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    train_g_op = tf.group([g_optim, update_ops])
    train_d_op = tf.group([d_optim, update_ops])
    '''
    
    '''
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    '''
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.gpu_fraction)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    
    train_writer = tf.summary.FileWriter('train_log', sess.graph)
    summary_op = tf.summary.merge_all()
    saver = tf.train.Saver()

    with tf.device('/device:GPU:0'):

        data_dir = 'C:\\Users\\Razer\\Downloads\\dataset'
        dataloader = cartoon_loader(args.patch_size, args.batch_size, data_dir)

        sess.run(tf.global_variables_initializer())
        total_iter = 0
        for epoch in range(args.total_epoch):
            '''
            if np.mod(epoch+1, 50) == 0:
                init_lr *= 0.5
            '''
            for batch in tqdm(dataloader):
                total_iter += 1
                for _ in range(3):
                    _, g_loss = sess.run([g_optim, g_loss_total], 
                                        feed_dict={input_image: batch[0], ref_image: batch[1], 
                                                    learning_rate: args.learning_rate})

                _, d_loss, train_info = sess.run([d_optim, d_loss_total, summary_op], 
                                                feed_dict={input_image: batch[0], ref_image: batch[1], 
                                                            learning_rate: args.learning_rate})    
                
                train_writer.add_summary(train_info, total_iter)

                if np.mod(total_iter, 50) == 0:
                    print('epoch: {}, iter: {}, d_loss: {}, g_loss: {}'\
                        .format(epoch, total_iter, d_loss, g_loss))
                    if np.mod(total_iter, 500) == 0:
                        saver.save(sess, 'saved_models/model', global_step=total_iter)


                
            
if __name__ == '__main__':
    args = arg_parser()
    train(args)
   