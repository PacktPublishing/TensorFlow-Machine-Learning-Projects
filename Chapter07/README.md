# Credit Card Fraud Detection Using Autoencoders 

<p>This project illustrates how to use autoencoders to classify fraudulent transactions in a credit card transaction data from Kaggle.</p>


### Installations
* This code is checked on using native Python 3 with anaconda
* Create a conda virtual environment and install the requirements using requirements.txt in the repo.
* We use Keras with Tensorflow backend for this project


#### Dataset
The dataset was downloaded from [Kaggle](https://www.kaggle.com/mlg-ulb/creditcardfraud). PLEASE download the data to "./data" folder as a ZIP file from the website before executing the code. Otherwise the code will run into errors. 


### Python Code Run Instructions
Make sure you have downloaded the dataset as per the instructions above. Then, To run the code just execute
```
python main.py
```


### Code Details
Code is pretty self explanatory. There are mainly four files in implementation:

* main.py :  Main function which runs the entire code
* model.py : Contains the model class which defines and trains the model
* utils.py : Contains general utility functions to load data and generate relevant plots
* parameters.py : Defines the static parameters used for building the model and storing the results


