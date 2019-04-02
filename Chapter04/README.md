# Digit Classification using Tensorflow Lite
This folder contains the code to build a deep learning model on MNIST handwritten digit dataset and converting the trained model to TF Lite format.

### Dataset
* https://www.tensorflow.org/guide/datasets

### Installations 
* This code is checked on using native Python 3 with anaconda
* Create a conda virtual environment and install relevant packages using requirements.txt file 
```
pip install requirements.txt
```
### Python Code Run Instructions
To run the code just execute 
```
python main.py
```
On CPU the code might take few mins to run. However, if you use GPUs it should be much faster
#### Dataset
Dataset used for this dataset is the standard MNIST handwritten digits dataset available in Tensorflow Datasets.

### Code Details
Code is pretty self explanatory. There are mainly 3 files in implementation:
* main.py  -- Implements the main function and also implements model building and training routines. 
* parameters.py -- Defines the parameters used in the code.
* utils.py -- Contains the helper functions for the code 

This code implements graph freezing and optimizing but will have to use Tensorflow Optimization Converter Tool (toco) to convert
the optimized graph to .tflite format. 

