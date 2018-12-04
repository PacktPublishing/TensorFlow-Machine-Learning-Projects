from parameter_config import *
from helper_functions import routing, squash



class CapsNet:
    def __init__(self):
        with tf.variable_scope('Input'):
            self.X = tf.placeholder(shape=[None, IMG_WIDTH, IMG_HEIGHT, N_CHANNELS], dtype=tf.float32, name="X")
            self.Y = tf.placeholder(shape=[None, N_CLASSES], dtype=tf.float32, name="Y")
            self.mask_with_labels = tf.placeholder_with_default(False, shape=(), name="mask_with_labels")

        self.define_network()
        self.define_loss()
        self.define_accuracy()
        self.define_optimizer()
        self.summary_()

    def define_network(self):
        with tf.variable_scope('Conv1_layer'):
            conv1_layer = tf.layers.conv2d(self.X, name="conv1_layer", **CONV1_LAYER_PARAMS) # [batch_size, 20, 20, 256]

        with tf.variable_scope('PrimaryCaps_layer'):
            conv2_layer = tf.layers.conv2d(conv1_layer, name="conv2_layer", **CONV2_LAYER_PARAMS) # [batch_size, 6, 6, 256]

            primary_caps = tf.reshape(conv2_layer, (BATCH_SIZE, NCAPS_CAPS1, CAPS_DIM_CAPS1, 1), name="primary_caps") # [batch_size, 1152, 8, 1]
            primary_caps_output = squash(primary_caps, name="caps1_output")
            # [batch_size, 1152, 8, 1]

        # DigitCaps layer, return [batch_size, 10, 16, 1]
        with tf.variable_scope('DigitCaps_layer'):
            digitcaps_input = tf.reshape(primary_caps_output, shape=(BATCH_SIZE, NCAPS_CAPS1, 1, CAPS_DIM_CAPS1, 1)) # [batch_size, 1152, 1, 8, 1]
            # [batch_size, 1152, 10, 1, 1]
            self.digitcaps_output = routing(digitcaps_input) # [batch_size, 10, 16, 1]

        # Decoder
        with tf.variable_scope('Masking'):
            self.v_norm = tf.sqrt(tf.reduce_sum(tf.square(self.digitcaps_output), axis=2, keep_dims=True) + tf.keras.backend.epsilon())

            predicted_class = tf.to_int32(tf.argmax(self.v_norm, axis=1)) #[batch_size, 10,1,1]
            self.y_predicted = tf.reshape(predicted_class, shape=(BATCH_SIZE,))  #[batch_size]
            y_predicted_one_hot = tf.one_hot(self.y_predicted, depth=NCAPS_CAPS2)  #[batch_size,10]  One hot operation

            reconstruction_targets = tf.cond(self.mask_with_labels,  # condition
                                      lambda: self.Y,  # if True (Training)
                                      lambda: y_predicted_one_hot,  # if False (Test)
                                      name="reconstruction_targets")

            digitcaps_output_masked = tf.multiply(tf.squeeze(self.digitcaps_output), tf.expand_dims(reconstruction_targets, -1)) # [batch_size, 10, 16]


            #Flattening as suggested by the paper
            decoder_input = tf.reshape(digitcaps_output_masked, [BATCH_SIZE, -1]) # [batch_size, 160]


        with tf.variable_scope('Decoder'):
            fc1 = tf.layers.dense(decoder_input, layer1_size, activation=tf.nn.relu, name="FC1") # [batch_size, 512]
            fc2 = tf.layers.dense(fc1, layer2_size, activation=tf.nn.relu, name="FC2") # [batch_size, 1024]
            self.decoder_output = tf.layers.dense(fc2, output_size, activation=tf.nn.sigmoid, name="FC3") # [batch_size, 784]


    def define_loss(self):
        # Margin Loss
        with tf.variable_scope('Margin_Loss'):
            # max(0, m_plus-||v_c||)^2
            positive_error = tf.square(tf.maximum(0., 0.9 - self.v_norm)) # [batch_size, 10, 1, 1]
            # max(0, ||v_c||-m_minus)^2
            negative_error = tf.square(tf.maximum(0., self.v_norm - 0.1)) # [batch_size, 10, 1, 1]
            # reshape: [batch_size, 10, 1, 1] => [batch_size, 10]
            positive_error = tf.reshape(positive_error, shape=(BATCH_SIZE, -1))
            negative_error = tf.reshape(negative_error, shape=(BATCH_SIZE, -1))

            Loss_vec = self.Y * positive_error + 0.5 * (1- self.Y) * negative_error # [batch_size, 10]
            self.margin_loss = tf.reduce_mean(tf.reduce_sum(Loss_vec, axis=1), name="margin_loss")

        # Reconstruction Loss
        with tf.variable_scope('Reconstruction_Loss'):
            ground_truth = tf.reshape(self.X, shape=(BATCH_SIZE, -1))
            self.reconstruction_loss = tf.reduce_mean(tf.square(self.decoder_output - ground_truth))

        # Combined Loss
        with tf.variable_scope('Combined_Loss'):
            self.combined_loss = self.margin_loss + 0.0005 * self.reconstruction_loss

    def define_accuracy(self):
        with tf.variable_scope('Accuracy'):
            correct_predictions = tf.equal(tf.to_int32(tf.argmax(self.Y, axis=1)), self.y_predicted)
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))

    def define_optimizer(self):
        with tf.variable_scope('Optimizer'):
            optimizer = tf.train.AdamOptimizer()
            self.train_optimizer = optimizer.minimize(self.combined_loss, name="training_optimizer")

    def summary_(self):
        reconstructed_image = tf.reshape(self.decoder_output, shape=(BATCH_SIZE, IMG_WIDTH, IMG_HEIGHT, N_CHANNELS))
        summary_list = [tf.summary.scalar('Loss/margin_loss', self.margin_loss),
                        tf.summary.scalar('Loss/reconstruction_loss', self.reconstruction_loss),
                        tf.summary.image('original', self.X),
                        tf.summary.image('reconstructed', reconstructed_image)]
        self.summary_ = tf.summary.merge(summary_list)
