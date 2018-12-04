from parameters import *
import tensorflow as tf
from tensorflow.contrib import seq2seq

class Model():
    def __init__(self, int_to_vocab):
        self.vocab_size = len(int_to_vocab)

        with tf.variable_scope('Input'):
            self.X = tf.placeholder(tf.int32, [None, None], name='input')
            self.Y = tf.placeholder(tf.int32, [None, None], name='target')
            self.input_shape = tf.shape(self.X)

        self.define_network()
        self.define_loss()
        self.define_optimizer()

    def define_network(self):
        # Define an init cell of RNN
        with tf.variable_scope("Network"):
            # Defining an initial cell state
            lstm = tf.contrib.rnn.BasicLSTMCell(RNN_SIZE)
            cell = tf.contrib.rnn.MultiRNNCell([lstm] * 2)  # Defining two LSTM layers for this case
            self.initial_state = cell.zero_state(self.input_shape[0], tf.float32)
            self.initial_state = tf.identity(self.initial_state, name="initial_state")

            embedding = tf.Variable(tf.random_uniform((self.vocab_size, RNN_SIZE), -1, 1))
            embed = tf.nn.embedding_lookup(embedding, self.X)

            outputs, self.final_state = tf.nn.dynamic_rnn(cell, embed, initial_state=None, dtype=tf.float32)
            self.final_state = tf.identity(self.final_state, name='final_state')
            self.predictions = tf.contrib.layers.fully_connected(outputs, self.vocab_size, activation_fn=None)
            # Probabilities for generating words
            probs = tf.nn.softmax(self.predictions, name='probs')

    def define_loss(self):
        # Defining the sequence loss
        with tf.variable_scope('Sequence_Loss'):
            self.loss = seq2seq.sequence_loss(self.predictions, self.Y,
                                              tf.ones([self.input_shape[0], self.input_shape[1]]))

    def define_optimizer(self):
        with tf.variable_scope("Optimizer"):
            optimizer = tf.train.AdamOptimizer(LEARNING_RATE)
            # Gradient Clipping
            gradients = optimizer.compute_gradients(self.loss)
            capped_gradients = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gradients]
            self.train_op = optimizer.apply_gradients(capped_gradients)