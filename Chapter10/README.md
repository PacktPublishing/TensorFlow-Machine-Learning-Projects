# Capsule Networks
This repo consists of an implementation of Capsule Networks as demonstrated by this paper [Dynamic Routing Between Capsules](https://arxiv.org/abs/1710.09829) on Fashion MNIST dataset.
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
On CPU the code might take few hours to run. However, if you use GPUs it should be much faster
#### Dataset
The dataset used for this illustration is [Fashion MNIST](https://github.com/zalandoresearch/fashion-mnist)

### Code Details
Code is pretty self explanatory. There are mainly four files in implementation:
* main.py  -- It contains three parts:
    * Train:  To train the model
    * Test:   Test the model on testing dataset
    * Visualize: Visualize few reconstructed images for further interpretation
* parameter_config.py -- Contains all the parameter declarations
* CapsNet.py -- Capsule network implementation. 
* helper_functions.py -- Helper functions. Also contains the **squashing** and **routing** functions  needed by capsule networks

Note that the model was not tuned for best hyperparameters. Feel free to play around.
