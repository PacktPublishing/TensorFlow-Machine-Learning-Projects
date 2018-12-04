import os
from parameter_config import *
import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('agg')


## Functions for Capsule Networks
def squash(vectors, name=None):
    """
    Squashing Function as implemented in the paper
    :parameter vectors: vector input that needs to be squashed
    :parameter name: Name of the tensor on the graph
    :return: a tensor with same shape as vectors but squashed as mentioned in the paper
    """
    with tf.name_scope(name, default_name="squash_op"):
        s_squared_norm = tf.reduce_sum(tf.square(vectors), axis=-2, keepdims=True)
        scale = s_squared_norm / (1. + s_squared_norm) / tf.sqrt(s_squared_norm + tf.keras.backend.epsilon())
        return scale*vectors


def routing(u):
    """
    This function performs the routing algorithm as mentioned in the paper
    :parameter u: Input tensor with [batch_size, num_caps_input_layer=1152, 1, caps_dim_input_layer=8, 1] shape.
                NCAPS_CAPS1: num capsules in the PrimaryCaps layer l
                CAPS_DIM_CAPS2: dimensions of output vectors of Primary caps layer l

    :return: "v_j" vector (tensor) in Digitcaps Layer
             Shape:[batch_size, NCAPS_CAPS1=10, CAPS_DIM_CAPS2=16, 1]
    """

    #local variable b_ij: [batch_size, num_caps_input_layer=1152, num_caps_output_layer=10, 1, 1]
                #num_caps_output_layer: number of capsules in Digicaps layer l+1
    b_ij = tf.zeros([BATCH_SIZE, NCAPS_CAPS1, NCAPS_CAPS2, 1, 1], dtype=np.float32, name="b_ij")

    # Preparing the input Tensor for total number of DigitCaps capsule for multiplication with W
    u = tf.tile(u, [1, 1, b_ij.shape[2].value, 1, 1])   # u => [batch_size, 1152, 10, 8, 1]


    # W: [num_caps_input_layer, num_caps_output_layer, len_u_i, len_v_j] as mentioned in the paper
    W = tf.get_variable('W', shape=(1, u.shape[1].value, b_ij.shape[2].value, u.shape[3].value, CAPS_DIM_CAPS2),
                        dtype=tf.float32, initializer=tf.random_normal_initializer(stddev=STDEV))
    W = tf.tile(W, [BATCH_SIZE, 1, 1, 1, 1]) # W => [batch_size, 1152, 10, 8, 16]

    #Computing u_hat (as mentioned in the paper)
    u_hat = tf.matmul(W, u, transpose_a=True)  # [batch_size, 1152, 10, 16, 1]

    # In forward, u_hat_stopped = u_hat;
    # In backward pass, no gradient pass from  u_hat_stopped to u_hat
    u_hat_stopped = tf.stop_gradient(u_hat, name='gradient_stop')

    # Routing Algorithm Begins here
    for r in range(ROUTING_ITERATIONS):
        with tf.variable_scope('iterations_' + str(r)):
            c_ij = tf.nn.softmax(b_ij, axis=2) # [batch_size, 1152, 10, 1, 1]

            # At last iteration, use `u_hat` in order to back propagate gradient
            if r == ROUTING_ITERATIONS - 1:
                s_j = tf.multiply(c_ij, u_hat) # [batch_size, 1152, 10, 16, 1]
                # then sum as per paper
                s_j = tf.reduce_sum(s_j, axis=1, keep_dims=True) # [batch_size, 1, 10, 16, 1]

                v_j = squash(s_j) # [batch_size, 1, 10, 16, 1]

            elif r < ROUTING_ITERATIONS - 1:  # No backpropagation in these iterations
                s_j = tf.multiply(c_ij, u_hat_stopped)
                s_j = tf.reduce_sum(s_j, axis=1, keepdims=True)
                v_j = squash(s_j)
                v_j = tf.tile(v_j, [1, u.shape[1].value, 1, 1, 1]) # [batch_size, 1152, 10, 16, 1]

                # Multiplying in last two dimensions: [16, 1]^T x [16, 1] yields [1, 1]
                u_hat_dot_v = tf.matmul(u_hat_stopped, v_j, transpose_a=True) # [batch_size, 1152, 10, 1, 1]

                b_ij = tf.add(b_ij,u_hat_dot_v)
    return tf.squeeze(v_j, axis=1) # [batch_size, 10, 16, 1]



def load_data(load_type='train'):
    '''

    :param load_type: train or test depending on the use case
    :return: x (images), y(labels)
    '''
    data_dir = os.path.join('data','fashion-mnist')
    if load_type == 'train':
        image_file = open(os.path.join(data_dir,'train-images-idx3-ubyte'))
        image_data = np.fromfile(file=image_file, dtype=np.uint8)
        x = image_data[16:].reshape((60000, 28, 28, 1)).astype(np.float32)

        label_file = open(os.path.join(data_dir, 'train-labels-idx1-ubyte'))
        label_data = np.fromfile(file=label_file, dtype=np.uint8)
        y = label_data[8:].reshape(60000).astype(np.int32)

        x_train = x[:55000] / 255.
        y_train = y[:55000]
        y_train = (np.arange(N_CLASSES) == y_train[:, None]).astype(np.float32)

        x_valid = x[55000:, ] / 255.
        y_valid = y[55000:]
        y_valid = (np.arange(N_CLASSES) == y_valid[:, None]).astype(np.float32)
        return x_train, y_train, x_valid, y_valid
    elif load_type == 'test':
        image_file = open(os.path.join(data_dir, 't10k-images-idx3-ubyte'))
        image_data = np.fromfile(file=image_file, dtype=np.uint8)
        x_test = image_data[16:].reshape((10000, 28, 28, 1)).astype(np.float)

        label_file = open(os.path.join(data_dir, 't10k-labels-idx1-ubyte'))
        label_data = np.fromfile(file=label_file, dtype=np.uint8)
        y_test = label_data[8:].reshape(10000).astype(np.int32)
        y_test = (np.arange(N_CLASSES) == y_test[:, None]).astype(np.float32)
        return x_test / 255., y_test


def shuffle_data(x, y):
    """ Shuffle the features and labels of input data"""
    perm = np.arange(y.shape[0])
    np.random.shuffle(perm)
    shuffle_x = x[perm,:,:,:]
    shuffle_y = y[perm]
    return shuffle_x, shuffle_y

def write_progress(op_type = 'train'):
    """
    Creating the handles for saving the results in a .csv file
    :return: appropriate logging files
    """
    if not os.path.exists(RESULTS_DIR):
        os.mkdir(RESULTS_DIR)
    if op_type == 'train':
        train_path = RESULTS_DIR  + '/' + 'train.csv'
        val_path = RESULTS_DIR + '/' + 'validation.csv'

        if os.path.exists(train_path):
            os.remove(train_path)
        if os.path.exists(val_path):
            os.remove(val_path)

        train_file = open(train_path, 'w')
        train_file.write('step,accuracy,loss\n')
        val_file = open(val_path, 'w')
        val_file.write('epoch,accuracy,loss\n')
        return train_file, val_file
    else:
        test_path = RESULTS_DIR + '/test.csv'
        if os.path.exists(test_path):
            os.remove(test_path)
        test_file = open(test_path, 'w')
        test_file.write('accuracy,loss\n')
        return test_file


def load_existing_details():
    """
    This function loads the train and val files to continue training
    :return: handles to train and val files and minimum validation loss
    """
    train_path = RESULTS_DIR  + '/' + 'train.csv'
    val_path = RESULTS_DIR + '/' + 'validation.csv'
    # finding the minimum validation loss so far
    f_val = open(val_path, 'r')
    lines = f_val.readlines()
    data = np.genfromtxt(lines[-1:], delimiter=',')
    min_loss = np.min(data[1:, 2])
    # loading the train and val files to continue training
    train_file = open(train_path, 'a')
    val_file = open(val_path, 'a')
    return train_file, val_file, min_loss


def eval_performance(sess, model, x, y):
    '''
    This function is mainly used to evaluate the accuracy on test and validation sets
    :param sess: session
    :param model: model to be used
    :param x: images
    :param y: labels
    :return: returns the average accuracy and loss for the dataset
    '''
    acc_all = loss_all = np.array([])
    num_batches = int(y.shape[0] / BATCH_SIZE)
    for batch_num in range(num_batches):
        start = batch_num * BATCH_SIZE
        end = start + BATCH_SIZE
        x_batch, y_batch = x[start:end], y[start:end]
        acc_batch, loss_batch, prediction_batch = sess.run([model.accuracy, model.combined_loss, model.y_predicted],
                                                     feed_dict={model.X: x_batch, model.Y: y_batch})
        acc_all = np.append(acc_all, acc_batch)
        loss_all = np.append(loss_all, loss_batch)
    return np.mean(acc_all), np.mean(loss_all)

def reconstruction(x, y, decoder_output, y_pred, n_samples):
    '''
    This function is used to reconstruct sample images for analysis
    :param x: Images
    :param y: Labels
    :param decoder_output: output from decoder
    :param y_pred: predictions from the model
    :param n_samples: num images
    :return: saves the reconstructed images
    '''

    sample_images = x.reshape(-1, IMG_WIDTH, IMG_HEIGHT)
    decoded_image = decoder_output.reshape([-1, IMG_WIDTH, IMG_WIDTH])

    fig = plt.figure(figsize=(n_samples * 2, 3))
    for i in range(n_samples):
        plt.subplot(1, n_samples, i+ 1)
        plt.imshow(sample_images[i], cmap="binary")
        plt.title("Label:" + IMAGE_LABELS[np.argmax(y[i])])
        plt.axis("off")
    fig.savefig(RESULTS_DIR + '/' + 'input_images.png')
    plt.show()

    fig = plt.figure(figsize=(n_samples * 2, 3))
    for i in range(n_samples):
        plt.subplot(1, n_samples, i + 1)
        plt.imshow(decoded_image[i], cmap="binary")
        plt.title("Prediction:" + IMAGE_LABELS[y_pred[i]])
        plt.axis("off")
    fig.savefig(RESULTS_DIR + '/' + 'decoder_images.png')
    plt.show()




