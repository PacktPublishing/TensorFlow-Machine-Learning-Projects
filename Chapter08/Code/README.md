# Traffic Sign Classification Using Bayesian Neural Networks
This repo consists of an implementation of a simple Bayesian Neural Networks using German Traffic Sign Dataset.

### Installations 
* This code is checked on using native Python 3 with anaconda
* Create a conda virtual environment and install relevant packages using requirements.txt file 
```
pip install requirements.txt
```
For installing Tensorflow Probability, use the following command:
```
pip install --upgrade tensorflow-probability

```

### Python Code Run Instructions
To run the code just execute 
```
python bnn.py
```

As the dataset is not large, execution should be fairly fast 
#### Dataset
We use the German Traffic Sign Dataset. You can download the dataset form [here](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset). On this website head over to Downloads section and then download "Images and Annotations" data from Training and Testing dataset subsections.
Also download,"Extended annotations including class ids" from  "Test Dataset" section in the website. 
Place the zip files under a folder named "Data" for code to execute.

### Code Details
Code is pretty self explanatory. There are mainly four files in implementation:
* bnn.py  -- Implements the main function and neural network model. It also contains training and inference parts of the model.
* parameters.py -- Defines the parameters used for modeling
* utils.py -- Implements utility functions to run the code. 

Note that the model was not tuned for best hyperparameters. Feel free to play around.
