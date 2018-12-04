# Gaussian Process Regression for Predicting Stock Prices

<p>This project illustrates how to use Gaussian Process to predict Stock Markets. We specifically stocks of Google, Netflix and GE as examples in this case.</p>


### Installations
* This code is checked on using native Python 3 with anaconda
* Create a conda virtual environment and install packages using requirements.txt
* We use plug and play functions from [GpFlow](https://github.com/GPflow/GPflow) library, which is a wrapper on top of Tensorflow for Gaussian Processes. Please install that library as mentioned in the README of that repo.




### Python Code Run Instructions
To run the code just execute
```
python main.py
```

#### Dataset
The dataset was downloaded from [Yahoo Finance](https://finance.yahoo.com). We downloaded the entire stock history for three companies:
* [Google] (https://finance.yahoo.com/quote/GOOG)
* [Netflix] (https://finance.yahoo.com/quote/NFLX)
* [General Electric Company] (https://finance.yahoo.com/quote/GE)  


### Code Details
Code is pretty self explanatory. There are mainly four files in implementation:

* main.py :  Main function which runs the entire code
* PreProcessing.py :  Preprocesses the stock data to make it ready for modeling
* VisualizeData.py : Contains the functions to visualize the dataset
* GP.py : Contains the implementation of training and inference through Gaussian Process using GpFlow library


