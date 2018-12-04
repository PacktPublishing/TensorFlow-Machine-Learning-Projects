'''
This file defines the parameters for the project
'''

## Directories

DATA_DIR = "./data"
PLOTS_DIR = "./plots"
MODEL_SAVE_DIR = "./saved_models"
LOG_DIR = "./logs"


## Training Parameters
RANDOM_SEED = 0
DIM_ENCODER = 14
EPOCHS = 100
BATCH_SIZE = 32
OPTIMIZER = 'adam'
LOSS = 'mean_squared_error'
EVAL_METRIC = 'accuracy'
