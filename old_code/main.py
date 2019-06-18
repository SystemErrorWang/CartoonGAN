import tensorflow as tf
import numpy as np
import argparse
import os
import time
import model
import utils


os.environ["CUDA_VISIBLE_DEVICES"]="0"

def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_size", default = 96, type = int)
    parser.add_argument("--crop_size", default = 16, type = int)
    parser.add_argument("--batch_size", default = 16, type = int)     
    parser.add_argument("--pre_train_iter", default = 20000, type = int)
    parser.add_argument("--iter", default = 100000, type = int)
    parser.add_argument("--learning_rate", default = 1e-4, type = float)
    parser.add_argument("--gpu_fraction", default = 0.5, type = float)
    parser.add_argument("--save_dir", default = 'saved_models')
    parser.add_argument("--train_out_dir", default = 'train_output')
    parser.add_argument("--test_out_dir", default = 'test_output')
    parser.add_argument("--mode", default = 'train')
    
    args = parser.parse_args()
    return args


class CartoonGAN():
    def __init__(self, args):
        self.image_size = args.image_size
        self.crop_size = args.crop_size
        self.batch_size = args.batch_size
        self.pre_train_iter = args.pre_train_iter
        self.iter = args.iter
        self.learning_rate = args.learning_rate
        self.gpu_fraction = args.gpu_fraction
        self.train_out_dir = args.train_out_dir
        self.test_out_dir = args.test_out_dir
        self.save_dir = args.save_dir
        self.lambda_ = 10
        
        self.is_train = tf.placeholder(tf.bool)
        self.photo_input = tf.placeholder(tf.float32, [None, None, None, 3], name="photo")
        self.cartoon_input = tf.placeholder(tf.float32, [None, None, None, 3], name="cartoon")
        self.blur_input = tf.placeholder(tf.float32, [None, None, None, 3], name="blur")


    
    def input_setup(self):
        
        self.celeba_list = utils.get_filename_list('celeba')
        self.cartoon_list = utils.get_filename_list('cartoon')
        print('Finished loading data')

            
    def build_model(self):
        
        self.fake_cartoon = model.generator(self.photo_input, name='generator', 
                                            reuse=False, is_train=self.is_train)

        self.real_logit_cartoon = model.multi_patch_discriminator(self.cartoon_input, self.crop_size, 
                                                            name='discriminator', reuse=False)    
        
        self.fake_logit_cartoon = model.multi_patch_discriminator(self.fake_cartoon, self.crop_size, 
                                                            name='discriminator', reuse=True)
        
        self.logit_blur = model.multi_patch_discriminator(self.blur_input, self.crop_size, 
                                                            name='discriminator', reuse=True)

        VGG_loss = utils.vgg_loss(self.photo_input, self.fake_cartoon)
        
        g_loss = -tf.reduce_mean(tf.log(tf.nn.sigmoid(self.fake_logit_cartoon))) + 5e3*VGG_loss
        
        d_loss = -tf.reduce_mean(tf.log(tf.nn.sigmoid(self.real_logit_cartoon))
                                + tf.log(1. - tf.nn.sigmoid(self.fake_logit_cartoon))
                                + tf.log(1. - tf.nn.sigmoid(self.logit_blur)))
  
        '''
        g_loss = -tf.reduce_mean(self.fake_logit_cartoon) + 10*VGG_loss

        d_loss = tf.reduce_mean(self.fake_logit_cartoon) - tf.reduce_mean(self.real_logit_cartoon)
        
        # Wasserstein loss with gradient penalty
        differences = self.fake_cartoon - self.cartoon_input
        alpha = tf.random_uniform(shape=[self.batch_size, self.image_size,
                                           self.image_size, 3], minval=0., maxval=1.)
        interpolates = self.cartoon_input + (alpha * differences)
        D_inter = model.patch_discriminator(interpolates, self.crop_size, 
                                      name='discriminator', reuse=True)
        gradients = tf.gradients(D_inter, [interpolates])[0]
        slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
        gradient_penalty = tf.reduce_mean((slopes - 1.) ** 2)
        d_loss += self.lambda_ * gradient_penalty
        '''

        all_vars = tf.trainable_variables()

        d_vars = [var for var in all_vars if 'discriminator' in var.name]
        g_vars = [var for var in all_vars if 'generator' in var.name]

        
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.init_optim = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=0., beta2=0.9).\
                                        minimize(VGG_loss, var_list=g_vars, colocate_gradients_with_ops=True)
            self.d_optim = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=0., beta2=0.9).\
                                        minimize(d_loss, var_list=d_vars, colocate_gradients_with_ops=True)
            self.g_optim = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=0., beta2=0.9).\
                                        minimize(g_loss, var_list=g_vars, colocate_gradients_with_ops=True)

        #Summary variables for tensorboard

        self.g_A_loss_summ = tf.summary.scalar('g_loss', g_loss)
        self.d_A_loss_summ = tf.summary.scalar('d_loss', d_loss)
        self.VGG_loss_summ = tf.summary.scalar('VGG_loss', VGG_loss)
        
        self.saver = tf.train.Saver(g_vars)
        
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=self.gpu_fraction)
        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        print('Finished building model')



    def train(self):
        if not os.path.exists(self.train_out_dir):
            os.makedirs(self.train_out_dir)
        
        # Initializing the global variables
        init = ([tf.global_variables_initializer(), tf.local_variables_initializer()])
        train_writer = tf.summary.FileWriter(self.save_dir+"/train", self.sess.graph)
        summary_op = tf.summary.merge_all()
        

        with tf.device('/device:GPU:0'):
            sess = self.sess
            sess.run(init)
            start_time = time.time()
            
            # Pre-training iterations
            if os.path.isfile(self.save_dir+ '/pre_train-{}.meta'.format(self.pre_train_iter-1)):
                self.saver.restore(sess, self.save_dir+ '/pre_train-'+str(self.pre_train_iter-1))
                print('Finished loading pre_trained model')
            else:
                for iter in range(self.pre_train_iter):
                    photo_batch = utils.next_batch(self.batch_size, self.image_size, self.celeba_list)
                    cartoon_batch, blur_batch = utils.next_blur_batch(self.batch_size, 
                                                                  self.image_size, 
                                                                  self.cartoon_list)
                
                    _ = sess.run([self.init_optim], feed_dict={self.photo_input: photo_batch, 
                                                                self.cartoon_input: cartoon_batch, 
                                                                self.blur_input: blur_batch, 
                                                                self.is_train: True})
    
                    if np.mod(iter+1, 50) == 0:
                        print('pre_train iteration:[%d/%d], time cost:%f' \
                                %(iter+1, self.pre_train_iter, time.time()-start_time))
                        start_time = time.time()

                        if np.mod(iter+1, 1000) == 0:
                            batch_image = sess.run([self.fake_cartoon], 
                                         feed_dict={self.photo_input: photo_batch, self.is_train: True})
                            batch_image = np.squeeze(batch_image)
                            utils.print_fused_image(batch_image, self.train_out_dir, str(iter)+'_pre_train.png', 4)
                        
                        if np.mod(iter+1, self.pre_train_iter) == 0:
                            self.saver.save(sess, self.save_dir+ '/pre_train', global_step=iter)
                        
            #Training iterations
            for iter in range(self.iter):                
                
                photo_batch = utils.next_batch(self.batch_size, self.image_size, self.celeba_list)
                cartoon_batch, blur_batch = utils.next_blur_batch(self.batch_size, 
                                                                  self.image_size, 
                                                                  self.cartoon_list)
                
                
                _ = sess.run([self.g_optim], feed_dict={self.photo_input: photo_batch, 
                                                        self.cartoon_input: cartoon_batch, 
                                                        self.blur_input: blur_batch, 
                                                        self.is_train: True})

                _, summary = sess.run([self.d_optim, summary_op], 
                                      feed_dict={self.photo_input: photo_batch, 
                                                self.cartoon_input: cartoon_batch, 
                                                self.blur_input: blur_batch, 
                                                self.is_train: True})      

                train_writer.add_summary(summary, iter)         
                    
                if np.mod(iter+1, 10) == 0:
                    print('train iteration:[%d/%d], time cost:%f' \
                            %(iter+1, self.iter, time.time()-start_time))
                    start_time = time.time()

                    if np.mod(iter+1, 500) == 0:
                        batch_image = sess.run([self.fake_cartoon], 
                                               feed_dict={self.photo_input: photo_batch, 
                                                          self.is_train: True})
                        batch_image = np.squeeze(batch_image)
                        utils.print_fused_image(batch_image, self.train_out_dir, str(iter)+'.png', 4 )
                        
                    if np.mod(iter+1, 20000) == 0:
                        self.saver.save(sess, self.save_dir+ '/model', global_step=iter)

    def test(self):
        
        if not os.path.exists(self.test_out_dir):
            os.mkdir(self.test_out_dir)
        
        self.test_list = utils.get_filename_list('actress')
        
        init = ([tf.global_variables_initializer(), tf.local_variables_initializer()])
        self.sess.run(init)
        self.saver.restore(self.sess, tf.train.latest_checkpoint(self.save_dir)) 

        for idx in range(100):
            photo_batch = utils.next_batch(self.batch_size, self.image_size, self.test_list)
            images = self.sess.run([self.fake_cartoon], feed_dict={self.photo_input: photo_batch, 
                                                                    self.is_train: True})
            images = np.squeeze(images)
            utils.print_fused_image(images, self.test_out_dir, str(idx)+'.png', 4 )



def main():
    args = arg_parser()
    model = CartoonGAN(args)
    
    if args.mode == 'train':
        model.build_model()
        model.input_setup()
        model.train()

    elif args.mode == 'test':
        model.build_model()
        model.test()
    
if __name__ == '__main__':

    main()