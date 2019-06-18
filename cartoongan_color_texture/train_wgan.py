import os
import tensorflow as tf
import numpy as np
import argparse
import network 
import utils
from tqdm import tqdm


os.environ["CUDA_VISIBLE_DEVICES"]="0"


def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--patch_size", default = 96, type = int)
    parser.add_argument("--batch_size", default = 16, type = int)     
    parser.add_argument("--pre_iter", default = 10000, type = int)
    parser.add_argument("--total_iter", default = 100000, type = int)
    parser.add_argument("--pre_train_lr", default = 1e-4, type = float)
    parser.add_argument("--adv_train_lr", default = 1e-4, type = float)
    parser.add_argument("--gpu_fraction", default = 0.5, type = float)
    parser.add_argument("--save_model_dir", default = 'saved_models')
    parser.add_argument("--save_out_dir", default = 'saved_results')
    parser.add_argument("--train_log_dir", default = 'train_log')
    parser.add_argument("--mode", default = 'train')
    
    args = parser.parse_args()
    
    return args
   
def train(args):
    

    input_photo = tf.placeholder(tf.float32, [args.batch_size, 
                                args.patch_size, args.patch_size, 3])
    input_cartoon = tf.placeholder(tf.float32, [args.batch_size, 
                                args.patch_size, args.patch_size, 3])
    is_training = tf.placeholder(tf.bool)
    
    generated_cartoon = network.generator_bn(input_photo, is_training=is_training)
    #generated_cartoon = network.generator(input_photo)
    
    blur_fake = utils.blur(generated_cartoon)
    blur_cartoon = utils.blur(input_cartoon)
    
    gray_fake = tf.image.rgb_to_grayscale(generated_cartoon)
    gray_cartoon = tf.image.rgb_to_grayscale(input_cartoon)
    
    d_loss_blur, g_loss_blur = utils.wgan_loss(network.disc_wgan, real=blur_cartoon, 
                                         fake=blur_fake, name='disc_blur')
    d_loss_gray, g_loss_gray = utils.wgan_loss(network.disc_wgan, real=gray_cartoon, 
                                         fake=gray_fake, name='disc_gray')
    
    real_blur_logit = network.discriminator_bn(blur_cartoon, is_training, reuse=False, name='disc_blur')
    fake_blur_logit = network.discriminator_bn(blur_fake, is_training, reuse=True, name='disc_blur')
    
    real_gray_logit = network.discriminator_bn(gray_cartoon, is_training, reuse=False, name='disc_gray')
    fake_gray_logit = network.discriminator_bn(gray_fake, is_training, reuse=True, name='disc_gray')
   
    vgg44_loss = utils.vggloss_4_4(generated_cartoon, input_photo)
    pixel_loss = tf.reduce_mean(tf.losses.absolute_difference(generated_cartoon, input_photo))
    
    tv_h = tf.reduce_mean((generated_cartoon[:, 1:, :, :] - 
                           generated_cartoon[:, :args.patch_size - 1, :, :])**2)
    tv_w = tf.reduce_mean((generated_cartoon[:, :, 1:, :] - 
                           generated_cartoon[:, :, :args.patch_size - 1, :])**2)
    tv_loss = (tv_h + tv_w)/(3*args.patch_size**2)
    
    g_loss_gray = -tf.reduce_mean(tf.log(fake_gray_logit)) 
    g_loss_blur = -tf.reduce_mean(tf.log(fake_blur_logit)) 
    
    d_loss_gray = -tf.reduce_mean(tf.log(real_gray_logit)
                            + tf.log(1. - fake_gray_logit))
    d_loss_blur = -tf.reduce_mean(tf.log(real_blur_logit)
                            + tf.log(1. - fake_blur_logit))
                         
    g_loss_total = 1e3*tv_loss + 1*(g_loss_blur + g_loss_gray) + 5e3*vgg44_loss
    d_loss_total = d_loss_blur + d_loss_gray
    
    all_vars = tf.trainable_variables()
    gen_vars = [var for var in all_vars if 'generator' in var.name]
    disc_vars = [var for var in all_vars if 'disc' in var.name]
    

    tf.summary.scalar('tv_loss', vgg44_loss)
    tf.summary.scalar('content_loss', vgg44_loss)
    tf.summary.scalar('generator_loss', g_loss_total)
    tf.summary.scalar('discriminator_loss', d_loss_total)
    
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        
        init_optim = tf.train.AdamOptimizer(args.pre_train_lr)\
                                .minimize(pixel_loss, var_list=gen_vars)
                                
        g_optim = tf.train.AdamOptimizer(args.adv_train_lr)\
                                        .minimize(g_loss_total, var_list=gen_vars)
        
        d_optim = tf.train.AdamOptimizer(args.adv_train_lr)\
                                        .minimize(d_loss_total, var_list=disc_vars)

    '''
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    '''
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.gpu_fraction)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    
    
    train_writer = tf.summary.FileWriter('train_log', sess.graph)
    summary_op = tf.summary.merge_all()
    saver = tf.train.Saver(var_list=gen_vars)

    with tf.device('/device:GPU:0'):

        sess.run(tf.global_variables_initializer())
        cartoon_dir = 'C:\\Users\\Razer\\Downloads\\dataset\\cartoon_jpg'
        photo_dir = 'C:\\Users\\Razer\\Downloads\\dataset\\celeba_jpg'
        
        
        photo_list = utils.load_image_list(photo_dir)
        cartoon_list = utils.load_image_list(cartoon_dir)
 
        '''
        for idx in tqdm(range(args.pre_iter)):
            photo_batch = utils.next_batch(photo_list, args.batch_size, args.patch_size)
            cartoon_batch = utils.next_batch(cartoon_list, args.batch_size, args.patch_size)
            #print(np.shape(photo_batch), np.shape(cartoon_batch))
                
            _, vgg_loss, train_info = sess.run([init_optim, vgg44_loss, summary_op], 
                                               feed_dict={input_photo: photo_batch, 
                                                          input_cartoon: cartoon_batch,
                                                          is_training: True})
            train_writer.add_summary(train_info, idx)

            if np.mod(idx+1, 50) == 0:
                print('iter: {}, vgg_loss: {}'.format(idx, vgg_loss))
                if np.mod(idx+1, 500) == 0:
                    saver.save(sess, 'pretrain_models/model', global_step=idx)
                    batch_image = sess.run([generated_cartoon], 
                                         feed_dict={input_photo: photo_batch, 
                                                    input_cartoon: cartoon_batch,
                                                    is_training: False})
                    batch_image = np.squeeze(batch_image)
                    utils.write_batch_image(batch_image, args.save_out_dir, str(idx)+'_pretrain.png', 4)
        '''
        
        saver.restore(sess, tf.train.latest_checkpoint('pretrain_models'))
        for idx in tqdm(range(args.total_iter)):
            
            photo_batch = utils.next_batch(photo_list, args.batch_size, args.patch_size)
            cartoon_batch = utils.next_batch(cartoon_list, args.batch_size, args.patch_size)
            
            #for batch in tqdm(dataloader):

            _, g_loss, vgg_loss = sess.run([g_optim, g_loss_total, vgg44_loss], 
                                feed_dict={input_photo: photo_batch, 
                                           input_cartoon: cartoon_batch,
                                           is_training: True})

            _, d_loss, train_info = sess.run([d_optim, d_loss_total, summary_op], 
                                            feed_dict={input_photo: photo_batch, 
                                                       input_cartoon: cartoon_batch,
                                                       is_training: True})  
 
            train_writer.add_summary(train_info, idx+args.pre_iter)
            
            if np.mod(idx+1, 50) == 0:

                print('iter: {}, d_loss: {}, g_loss: {}, vgg_loss: {}'\
                      .format(idx, d_loss, g_loss, vgg_loss))
                if np.mod(idx+1, 500) == 0:
                    saver.save(sess, 'saved_models/model', global_step=idx)
                    
                    batch_image = sess.run([generated_cartoon], 
                                         feed_dict={input_photo: photo_batch, 
                                                    input_cartoon: cartoon_batch,
                                                    is_training: False})
                    batch_image = np.squeeze(batch_image)
                    utils.write_batch_image(batch_image, args.save_out_dir, str(idx)+'_train.png', 4)
        
 
            
if __name__ == '__main__':
    args = arg_parser()
    train(args)  
   
    
    

   