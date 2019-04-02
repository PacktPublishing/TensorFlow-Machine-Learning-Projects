# Sentiment Analysis Using Tensorflow Js
This folder consists of the code for Chapter 1 of the book. It creates a sentiment anaylsis model using the movie reviews dataset mentioned.

## Data
* https://www.kaggle.com/c/si650winter11/data

### Installations 
* This code is checked on using native Python 3.6 with Anaconda 
* Create a conda virtual environment and install relevant packages using requirements.txt file 
```
pip install requirements.txt
```
### Python Code Run Instructions
Navigate the terminal to the directory and execute the command in virtual environment
```
python main.py 
```
This will create the model 

### Run HTML File
Open the file Run_On_Browser.HTML in Chrome to run the model on browser. Note that you may need to configure your server to allow [Cross-Origin Resource Sharing (CORS)](https://enable-cors.org/), in order to allow fetching the files in JavaScript. Easy way to get CORS working is to install [Chrome CORS Extension](https://chrome.google.com/webstore/detail/allow-control-allow-origi/nlfbmbojpeacfghkpbjhddihlkkiljbi?hl=en).

Once CORS is enabled you need to start a python HTTP server for serving the model to the domain. This can be achieved by exectuing the following command in the directory of model json and other files.
```
python -m SimpleHTTPServer
```
Once the server is started you can go to browser and access the HTML file. Note that as soon as you load the file it will take 1-2 seconds to load the model and token index file. 
Once the system is ready, type in a review and click submit. It should print the entire Tensor of score at the Top (apologies for formatting). 

Typical Inputs to try:
* awesome movie
* Terrible movie
* that movie really sucks
* I like that movie
*  hate the movie

Play around and enjoy!!

#### TroubleShooting
Note that you might encounter an error while loading the model json in HTML. This might be because of names of layer mismatch.This is an ongoing issue with Keras and they are trying to fix it. For now, you might need to remove "gru_cell" part from all the layer names in the json manually. For e.g. "gru_1/gru_cell/biases" -> "gru_1/biases"
