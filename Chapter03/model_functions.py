#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
This file defines the model functions to be used for training

@author: ankit.jain
"""
import os
import pandas as pd
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, GRU, Embedding
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.models import save_model
from preprocess_functions import create_sequences


# Defining parameters 
BATCH_SIZE = 32
EPOCHS =15
VAL_SPLIT = 0.05 # Fraction of data to be used for validation
EMBEDDING_SIZE =8


def define_model(num_tokens,max_tokens):
    '''
    Defines the model definition based on input parameters
    '''
    model = Sequential()
    model.add(Embedding(input_dim=num_tokens,
                    output_dim=EMBEDDING_SIZE,
                    input_length=max_tokens,
                    name='layer_embedding'))

    model.add(GRU(units=16, name = "gru_1",return_sequences=True))
    model.add(GRU(units=8, name = "gru_2" ,return_sequences=True))
    model.add(GRU(units=4, name= "gru_3"))
    model.add(Dense(1, activation='sigmoid',name="dense_1"))
    optimizer = Adam(lr=1e-3)
    model.compile(loss='binary_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'])
    print model.summary()
    return model


def train_model(model,input_sequences,y_train):
    '''
    Train the model based on input parameters
    '''
    
    model.fit(input_sequences, y_train,
          validation_split=VAL_SPLIT, epochs=EPOCHS, batch_size=BATCH_SIZE)
    return model

def test_model(model,token_idx,max_tokens):
    '''
    Testing the model on sample data
    '''
    txt = ["awesome movie","Terrible movie","that movie really sucks","I like that movie","hate the movie"]
    pred = model.predict(create_sequences(txt,token_idx,max_tokens))
    pred = [pred[i][0] for i in range(len(txt))]
    output_df = pd.DataFrame({"Review Text": txt, "Prediction Score": pred})
    output_df = output_df.loc[:,['Review Text','Prediction Score']]
    
    print output_df

def model_save(model, output_dir):
    '''
    Saving the model
    '''
    output_file = os.path.join(output_dir,"sentiment_analysis_model.h5" )
    save_model(model,output_file,overwrite=True,include_optimizer=True)
    
    
    
