#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
This script takes the input data and trains a sentiment analysis model using neural networks
@author: ankit.jain
"""

from preprocess_functions import *  # Importing all preprocessing functions
from model_functions import *  # Importing all model functions
import tensorflow as tf
import os 
import tensorflowjs as tfjs


# PARAMETERS
current_dir = os.path.dirname(os.path.realpath(__file__))
INPUT_FILE = os.path.join(current_dir, "sentiment.txt")
OUTPUT_DIR = current_dir

def main():
    #Read and preprocessing the data 
    print "=== Read the input data ==="
    X_text , Y = get_data(INPUT_FILE)
    #Get the relevant token dict
    print " ===Tokenizing Reviews === "
    token_idx,num_tokens = tokenize_text(X_text)
    print 'Num of unique tokens are',num_tokens
    max_tokens = get_max(X_text)
    print "Max number of tokens in a review are", max_tokens
    print "=== Creating Input Sequences ==="
    input_sequences = create_sequences(X_text, token_idx,max_tokens)
    print "=== Defining the model ==="
    model = define_model(num_tokens,max_tokens)
    print "=== Training the model==="
    model = train_model(model,input_sequences,Y)
    print "=== Testing the model with some inputs and the output is ==="
    test_model(model,token_idx,max_tokens)
    print "=== Saving Model ==="
    tfjs.converters.save_keras_model(model, OUTPUT_DIR)    
    #model_save(model, OUTPUT_DIR)
    print "=== Saving the Token Index Dict for Tensorflow Js"
    create_csv(token_idx, 'token_index.csv',OUTPUT_DIR)


if __name__ == "__main__":
    main()
    
    
    
    
    
    
    
    



    