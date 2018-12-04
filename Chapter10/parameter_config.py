import argparse
import tensorflow as tf
import numpy as np

parser = argparse.ArgumentParser()

##### Arguments ####

## Fashion MNIST Parameters
N_CLASSES = 10
IMG_WIDTH = 28
IMG_HEIGHT = 28
N_CHANNELS = 1                 # Number of Input Channels
IMAGE_LABELS = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                            'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']



## Model Parameters
CONV1_LAYER_PARAMS =  {"filters": 256,
                "kernel_size": 9,
                "activation": tf.nn.relu,
                "padding": "valid",
                "strides": 1
                }

# Parameters of PrimaryCaps_layer
MAPS_CAPS1 = 32
NCAPS_CAPS1 = MAPS_CAPS1*6*6  # Total number of primary capsules = 1152
CAPS_DIM_CAPS1 = 8            # Dimensions of each capsule

CONV2_LAYER_PARAMS  = {"filters": MAPS_CAPS1 * CAPS_DIM_CAPS1,  # Total Convolutional Filters = 256
                "kernel_size": 9,
                "strides": 2,
                "padding": "valid",
                "activation": tf.nn.relu}

# Parameters of DigitCaps_layer
NCAPS_CAPS2 = 10
CAPS_DIM_CAPS2 = 16           # Dimension of each capsule in layer 2

# Decoder Parameters
layer1_size = 512
layer2_size = 1024
output_size = IMG_WIDTH* IMG_HEIGHT

## Loss

# Margin Loss
M_PLUS = 0.9
M_MINUS= 0.1
LAMBDA = 0.5

# Reconstruction Loss
ALPHA = 0.0005

# Training Params
BATCH_SIZE = 128
EPOCHS = 20
ROUTING_ITERATIONS = 3    # Routing Iterations
STDEV = 0.01  # STDEV for Weight Initialization


## Environment and Save Directories
RESTORE_TRAINING = False            # Restores the trained model
CHECKPOINT_PATH_DIR = './model_dir'
LOG_DIR = './logs/'
RESULTS_DIR = './results/'
STEPS_TO_SAVE = 100                 # Frequency (in steps) of saving the train result

## Visualization Parameters
N_SAMPLES = 3                       # No. of Samples Images to Save



