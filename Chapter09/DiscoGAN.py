import tensorflow as tf
from parameters import *
from utils import generator, discriminator
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'



class DiscoGAN:
    def __init__(self):
        with tf.variable_scope('Input'):
            self.X_bags = tf.placeholder(shape = [None, 64, 64, 3], name='bags', dtype=tf.float32)
            self.X_shoes = tf.placeholder(shape= [None, 64, 64, 3], name='shoes',dtype= tf.float32)
        self.initializer = tf.truncated_normal_initializer(stddev=0.02)
        self.define_network()
        self.define_loss()
        self.get_trainable_params()
        self.define_optimizer()
        self.summary_()

    def define_network(self):
        
        # Generators
        # This one is used to generate fake data
        self.gen_b_fake = generator(self.X_shoes, self.initializer,scope_name="generator_sb")
        self.gen_s_fake =   generator(self.X_bags, self.initializer,scope_name="generator_bs")

        # Reconstruction Generators
        # Note that parameters are being used from previous layers
        self.gen_recon_s = generator(self.gen_b_fake, self.initializer,scope_name="generator_sb",  reuse=True)
        self.gen_recon_b = generator(self.gen_s_fake,  self.initializer, scope_name="generator_bs", reuse=True)

        # Discriminator for Shoes
        self.disc_s_real = discriminator(self.X_shoes,self.initializer, scope_name="discriminator_s")
        self.disc_s_fake = discriminator(self.gen_s_fake,self.initializer, scope_name="discriminator_s", reuse=True)

        # Discriminator for Bags
        self.disc_b_real = discriminator(self.X_bags,self.initializer,scope_name="discriminator_b")
        self.disc_b_fake = discriminator(self.gen_b_fake, self.initializer, reuse=True,scope_name="discriminator_b")

        # Defining Discriminators of Bags and Shoes

    def define_loss(self):
        # Reconstruction loss for generators
        self.const_loss_s = tf.reduce_mean(tf.losses.mean_squared_error(self.gen_recon_s, self.X_shoes))
        self.const_loss_b = tf.reduce_mean(tf.losses.mean_squared_error(self.gen_recon_b, self.X_bags))

        # Generator loss for GANs
        self.gen_s_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=self.disc_s_fake, labels=tf.ones_like(self.disc_s_fake)))
        self.gen_b_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=self.disc_b_fake, labels=tf.ones_like(self.disc_b_fake)))

        # Total Generator Loss
        self.gen_loss =  (self.const_loss_b + self.const_loss_s)  + self.gen_s_loss + self.gen_b_loss

        # Cross Entropy loss for discriminators for shoes and bags
        # Shoes
        self.disc_s_real_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=self.disc_s_real, labels=tf.ones_like(self.disc_s_real)))
        self.disc_s_fake_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=self.disc_s_fake, labels=tf.zeros_like(self.disc_s_fake)))
        self.disc_s_loss = self.disc_s_real_loss + self.disc_s_fake_loss  # Combined


        # Bags
        self.disc_b_real_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=self.disc_b_real, labels=tf.ones_like(self.disc_b_real)))
        self.disc_b_fake_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=self.disc_b_fake, labels=tf.zeros_like(self.disc_b_fake)))
        self.disc_b_loss = self.disc_b_real_loss + self.disc_b_fake_loss

        # Total Discriminator Loss
        self.disc_loss = self.disc_b_loss + self.disc_s_loss

    def get_trainable_params(self):
        '''
        This function is useful for obtaining trainable parameters which need to be trained either with discriminator or generator loss
        :return:
        '''
        self.disc_params = []
        self.gen_params = []
        for var in tf.trainable_variables():
            if 'generator' in var.name:
                self.gen_params.append(var)
            elif 'discriminator' in var.name:
                self.disc_params.append(var)

    def define_optimizer(self):
        self.disc_optimizer = tf.train.AdamOptimizer(LEARNING_RATE).minimize(self.disc_loss, var_list=self.disc_params)
        self.gen_optimizer = tf.train.AdamOptimizer(LEARNING_RATE).minimize(self.gen_loss, var_list=self.gen_params)

    def summary_(self):
        # Store the losses
        tf.summary.scalar("gen_loss", self.gen_loss)
        tf.summary.scalar("gen_s_loss", self.gen_s_loss)
        tf.summary.scalar("gen_b_loss", self.gen_b_loss)
        tf.summary.scalar("const_loss_s", self.const_loss_s)
        tf.summary.scalar("const_loss_b", self.const_loss_b)
        tf.summary.scalar("disc_loss", self.disc_loss)
        tf.summary.scalar("disc_b_loss", self.disc_b_loss)
        tf.summary.scalar("disc_s_loss", self.disc_s_loss)

        # Histograms for all vars
        for var in tf.trainable_variables():
            tf.summary.histogram(var.name, var)

        self.summary_ = tf.summary.merge_all()



