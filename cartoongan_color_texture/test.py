import os
import cv2
import tensorflow as tf
import numpy as np
from network import generator, discriminator
from dataset import get_test_loader
from utils import vggloss_5_4, blur
from tqdm import tqdm


os.environ["CUDA_VISIBLE_DEVICES"]="0"


def test():
    
    input_image = tf.placeholder(tf.float32, [None, None, None, 3])
    result_image = generator(input_image, name='generator') 
    
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.99)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    saver = tf.train.Saver()

    
    if not os.path.exists('results'):
        os.mkdir('results')
        
    with tf.device('/device:GPU:0'):

        data_dir = 'C:\\Users\\Razer\\Downloads\\dataset\\dped'
        dataloader = get_test_loader(data_dir)

        sess.run(tf.global_variables_initializer())
        saver.restore(sess, tf.train.latest_checkpoint('saved_models'))
        
        for idx, batch in tqdm(enumerate(dataloader)):
            result = sess.run([result_image], feed_dict={input_image: batch})
            result = (np.squeeze(result)+1)*127.5
            result = np.clip(result, 0, 255).astype(np.float32)
            save_path = os.path.join('results', '{}.jpg'.format(str(idx).zfill(4)))
            cv2.imwrite(save_path, result)

                

                
            
if __name__ == '__main__':
    test()
   