'''
This file contains all the parameters required for running the model
'''


# Training Parameters
NUM_EPOCHS = 500
LEARNING_RATE = 0.001 # Learning Rate
BATCH_SIZE = 128 # Batch Size
CHECKPOINT_PATH_DIR = './model_dir'
RESTORE_TRAINING=False
SAVE_DIR = './save'

# Network Parameters
RNN_SIZE = 128 # RNN Size
SEQ_LENGTH = 32  # Sequence Length

# Data Parameters
TEXT_SAVE_DIR= "./data/postgre_book.txt"
