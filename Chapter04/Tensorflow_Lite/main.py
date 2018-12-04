from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D,Activation
from keras.callbacks import TensorBoard
from keras import backend as K

from utils import *


# Creating the relevant directories
MODEL_DIR = create_model_dir()
FREEZE_GRAPH_DIR =  create_freeze_graph_dir(MODEL_DIR)
OPTIMIZED_GRAPH_DIR =create_optimized_graph_dir(MODEL_DIR)



def prepare_training_data():
    # the data, split between train and test sets
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train.reshape(x_train.shape[0], IMAGE_SIZE, IMAGE_SIZE, 1)
    x_test = x_test.reshape(x_test.shape[0], IMAGE_SIZE, IMAGE_SIZE, 1)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'Num Training images')
    print(x_test.shape[0], 'Num Testing images')

    # Converting the target variable to categorical
    y_train = keras.utils.to_categorical(y_train, NUM_CLASSES)
    y_test = keras.utils.to_categorical(y_test, NUM_CLASSES)

    return x_train,x_test,y_train,y_test

class Model():
    def __init__(self):
        self.x_train, self.x_test,self.y_train, self.y_test = prepare_training_data()
        self.model = self.define_model()

    def define_model(self):
        model = Sequential()
        model.add(Conv2D(32, kernel_size=(3, 3),
                         activation='relu',
                         input_shape=INPUT_SHAPE))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.5))

        model.add(Dense(NUM_CLASSES))
        model.add(Activation('softmax', name = 'softmax_tensor'))

        model.compile(loss=keras.losses.categorical_crossentropy,
                      optimizer=keras.optimizers.Adadelta(),
                      metrics=['accuracy'])
        tensorboard = TensorBoard(log_dir=MODEL_DIR)
        self.model = model
        self.tensorboard = tensorboard

    def train_model(self):
        self.model.fit(self.x_train, self.y_train,
                  batch_size=BATCH_SIZE,
                  epochs=EPOCHS,
                  verbose=1,
                  validation_data=(self.x_test, self.y_test),
                       callbacks = [self.tensorboard])
        score = self.model.evaluate(self.x_test, self.y_test, verbose=0)
        print('Test loss:', score[0])
        print('Test accuracy:', score[1])

def main():

    # Load and prepare training data
    model = Model()
    print ("Defining Model")
    model.define_model()
    print ("Training Model")
    model.train_model()
    print ("Creating Frozen Graph")
    sess = K.get_session()
    create_frozen_graph(sess, ['softmax_tensor_1/Softmax'],FREEZE_GRAPH_DIR)

    print ("Converting Frozen Graph To Tensorboard compatible file")
    pb_to_tensorboard(FREEZE_GRAPH_DIR, "freeze")

    print ("Optimizing the graph for inference")
    optimize_graph(FREEZE_GRAPH_DIR,OPTIMIZED_GRAPH_DIR)
    pb_to_tensorboard(OPTIMIZED_GRAPH_DIR,"optimize")


if __name__ == "__main__":
    main()

