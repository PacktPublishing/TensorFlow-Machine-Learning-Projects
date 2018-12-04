'''
This file consists of the helper functions for processing
'''

## Package Imports
from PIL import Image
import os
from parameters import *
import tensorflow as tf
import numpy as np
import glob
try:
    import wget
except:
    print ("Can't import wget as you are probably on windows laptop")

def extract_files(data_dir,type = 'bags'):
    '''
    :param data_dir: Input directory
    :param type: bags or shoes
    :return: saves the cropped files to the bags to shoes directory
    '''
    input_file_dir = os.path.join(os.getcwd(),data_dir, "train")
    result_dir = os.path.join(os.getcwd(),type)
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    file_names= os.listdir(input_file_dir)
    for file in file_names:
        input_image = Image.open(os.path.join(input_file_dir,file))
        input_image = input_image.resize([128, 64])
        input_image = input_image.crop([64, 0, 128, 64])  # Cropping only the colored image. Excluding the edge image
        input_image.save(os.path.join(result_dir,file))


def generate_dataset():
    '''
    Before executing this function. Follow these steps;
    1. Download the datasets
    Handbags data Link 'https://people.eecs.berkeley.edu/~tinghuiz/projects/pix2pix/datasets/edges2handbags.tar.gz'
    Shoes data Link 'https://people.eecs.berkeley.edu/~tinghuiz/projects/pix2pix/datasets/edges2shoes.tar.gz'

    2. Extract the tar files.

    3. Execute this function. This function will extract the handbags and shoe images from the datasets.
    '''
    if not os.path.exists(os.path.join(os.getcwd(), "edges2handbags")):
        try:
            print ("Downloading dataset")
            bag_data_link = 'https://people.eecs.berkeley.edu/~tinghuiz/projects/pix2pix/datasets/edges2handbags.tar.gz'
            shoe_data_link = 'https://people.eecs.berkeley.edu/~tinghuiz/projects/pix2pix/datasets/edges2shoes.tar.gz'
    
            wget.download(bag_data_link)
            wget.download(shoe_data_link)
    
            with tarfile.open('./edges2handbags.tar.gz') as tar:
                tar.extractall()
                tar.close()
    
            with tarfile.open('./edges2shoes.tar.gz') as tar:
                tar.extractall()
                tar.close()
        except:
            print ("It seems you are on windows laptop. Please download the data as instructed in README before executing the code")

    extract_files("edges2handbags", 'bags')
    extract_files("edges2shoes", 'shoes')


def load_data(load_type = 'train'):
    shoelist = glob.glob(os.path.join(os.getcwd(), "shoes/*jpg"))
    shoe_data = np.array([np.array(Image.open(fname)) for fname in shoelist]).astype(np.float32)
    baglist = glob.glob(os.path.join(os.getcwd(), "bags/*jpg"))
    bags_data = np.array([np.array(Image.open(fname)) for fname in baglist]).astype(np.float32)
    shoe_data = shoe_data/255.
    bags_data = bags_data/255.
    return shoe_data, bags_data


def save_image(global_step, img_data, file_name):
    sample_results_dir = os.path.join(os.getcwd(), "sample_results", "epoch_" +str(global_step))
    if not os.path.exists(sample_results_dir):
        os.makedirs(sample_results_dir)


    result = Image.fromarray((img_data[0] * 255).astype(np.uint8))
    result.save(os.path.join(sample_results_dir, file_name + ".jpg"))



def discriminator(x,initializer, scope_name ='discriminator',  reuse=False):
    with tf.variable_scope(scope_name) as scope:
        if reuse:
            scope.reuse_variables()
        conv1 = tf.contrib.layers.conv2d(inputs=x, num_outputs=32, kernel_size=4, stride=2, padding="SAME",
                                         reuse=reuse, activation_fn=tf.nn.leaky_relu, weights_initializer=initializer,
                                         scope="disc_conv1")  # 32 x 32 x 32
        conv2 = tf.contrib.layers.conv2d(inputs=conv1, num_outputs=64, kernel_size=4, stride=2, padding="SAME",
                                         reuse=reuse, activation_fn=tf.nn.leaky_relu, normalizer_fn=tf.contrib.layers.batch_norm,
                                         weights_initializer=initializer, scope="disc_conv2")  # 16 x 16 x 64
        conv3 = tf.contrib.layers.conv2d(inputs=conv2, num_outputs=128, kernel_size=4, stride=2, padding="SAME",
                                         reuse=reuse, activation_fn=tf.nn.leaky_relu, normalizer_fn=tf.contrib.layers.batch_norm,
                                         weights_initializer=initializer, scope="disc_conv3")  # 8 x 8 x 128
        conv4 = tf.contrib.layers.conv2d(inputs=conv3, num_outputs=256, kernel_size=4, stride=2, padding="SAME",
                                         reuse=reuse, activation_fn=tf.nn.leaky_relu, normalizer_fn=tf.contrib.layers.batch_norm,
                                         weights_initializer=initializer, scope="disc_conv4")  # 4 x 4 x 256
        conv5 = tf.contrib.layers.conv2d(inputs=conv4, num_outputs=512, kernel_size=4, stride=2, padding="SAME",
                                         reuse=reuse, activation_fn=tf.nn.leaky_relu, normalizer_fn=tf.contrib.layers.batch_norm,
                                         weights_initializer=initializer, scope="disc_conv5")  # 2 x 2 x 512
        fc1 = tf.reshape(conv5, shape=[tf.shape(x)[0], 2 * 2 * 512])
        fc1 = tf.contrib.layers.fully_connected(inputs=fc1, num_outputs=512, reuse=reuse, activation_fn=tf.nn.leaky_relu,
                                                normalizer_fn=tf.contrib.layers.batch_norm,
                                                weights_initializer=initializer, scope="disc_fc1")
        fc2 = tf.contrib.layers.fully_connected(inputs=fc1, num_outputs=1, reuse=reuse, activation_fn=tf.nn.sigmoid,
                                                weights_initializer=initializer, scope="disc_fc2")

        return fc2


def generator(x, initializer, scope_name = 'generator',reuse=False):
    with tf.variable_scope(scope_name) as scope:
        if reuse:
            scope.reuse_variables()
        conv1 = tf.contrib.layers.conv2d(inputs=x, num_outputs=32, kernel_size=4, stride=2, padding="SAME",
                                         reuse=reuse, activation_fn=tf.nn.leaky_relu, weights_initializer=initializer,
                                         scope="disc_conv1")  # 32 x 32 x 32
        conv2 = tf.contrib.layers.conv2d(inputs=conv1, num_outputs=64, kernel_size=4, stride=2, padding="SAME",
                                         reuse=reuse, activation_fn=tf.nn.leaky_relu, normalizer_fn=tf.contrib.layers.batch_norm,
                                         weights_initializer=initializer, scope="disc_conv2")  # 16 x 16 x 64
        conv3 = tf.contrib.layers.conv2d(inputs=conv2, num_outputs=128, kernel_size=4, stride=2, padding="SAME",
                                         reuse=reuse, activation_fn=tf.nn.leaky_relu, normalizer_fn=tf.contrib.layers.batch_norm,
                                         weights_initializer=initializer, scope="disc_conv3")  # 8 x 8 x 128
        conv4 = tf.contrib.layers.conv2d(inputs=conv3, num_outputs=256, kernel_size=4, stride=2, padding="SAME",
                                         reuse=reuse, activation_fn=tf.nn.leaky_relu, normalizer_fn=tf.contrib.layers.batch_norm,
                                         weights_initializer=initializer, scope="disc_conv4")  # 4 x 4 x 256

        deconv1 = tf.contrib.layers.conv2d(conv4, num_outputs=4 * 128, kernel_size=4, stride=1, padding="SAME",
                                               activation_fn=tf.nn.relu, normalizer_fn=tf.contrib.layers.batch_norm,
                                               weights_initializer=initializer, scope="gen_conv1")
        deconv1 = tf.reshape(deconv1, shape=[tf.shape(x)[0], 8, 8, 128])

        deconv2 = tf.contrib.layers.conv2d(deconv1, num_outputs=4 * 64, kernel_size=4, stride=1, padding="SAME",
                                               activation_fn=tf.nn.relu, normalizer_fn=tf.contrib.layers.batch_norm,
                                               weights_initializer=initializer, scope="gen_conv2")
        deconv2 = tf.reshape(deconv2, shape=[tf.shape(x)[0], 16, 16, 64])

        deconv3 = tf.contrib.layers.conv2d(deconv2, num_outputs=4 * 32, kernel_size=4, stride=1, padding="SAME",
                                               activation_fn=tf.nn.relu, normalizer_fn=tf.contrib.layers.batch_norm,
                                               weights_initializer=initializer, scope="gen_conv3")
        deconv3 = tf.reshape(deconv3, shape=[tf.shape(x)[0], 32, 32, 32])

        deconv4 = tf.contrib.layers.conv2d(deconv3, num_outputs=4 * 16, kernel_size=4, stride=1, padding="SAME",
                                               activation_fn=tf.nn.relu, normalizer_fn=tf.contrib.layers.batch_norm,
                                               weights_initializer=initializer, scope="gen_conv4")
        deconv4 = tf.reshape(deconv4, shape=[tf.shape(x)[0], 64, 64, 16])

        recon = tf.contrib.layers.conv2d(deconv4, num_outputs=3, kernel_size=4, stride=1, padding="SAME", \
                                             activation_fn=tf.nn.relu, scope="gen_conv5")

        return recon
    