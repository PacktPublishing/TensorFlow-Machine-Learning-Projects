# Generating Book Scripts using LSTMs
This repo consists of an implementation of Book Script Generation code.

### Dataset
The dataset used was from one of the popular Packt book Mastering PostgreSQL by Hans-Jürgen Schönig. We used almost 100 pages from the book and removed any figures, tables and SQL code. 

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
As the dataset is not large, code can be executed on CPU itself.

### Code Details
Code is pretty self explanatory. There are mainly four files in implementation:
* main.py  -- It contains three parts:
    * main function:  To call the relevant functions
    * train:   trains the model
* parameters.py -- Contains all the parameter declarations
* Model.py -- Contains the Model Class.
* utils.py -- Helper functions.

Note that the model was not tuned for best hyperparameters. Feel free to play around.
