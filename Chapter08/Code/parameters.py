import os

#DATA_DIR = '/Users/ankit.jain/Documents/Teaching&Learning/Packt/Book/BNN/Data'
DATA_DIR = os.path.join(os.getcwd(),"..","Data")
NUM_CLASSES = 43
IMG_SIZE = 32

#Training Parameters
BATCH_SIZE =128
EPOCHS =1000
LEARNING_RATE = 0.001

# Inference Parameters
NUM_MONTE_CARLO = 50
