# DiscoGANs
This repo consists of an implementation of DiscoGANs as demonstrated by this paper [DiscoGANs](https://arxiv.org/pdf/1703.05192.pdf) on Handbags and Shoes dataset.

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
The dataset used for this project is
* [Handbags Dataset](https://people.eecs.berkeley.edu/~tinghuiz/projects/pix2pix/datasets/edges2handbags.tar.gz)
* [Shoes data](https://people.eecs.berkeley.edu/~tinghuiz/projects/pix2pix/datasets/edges2shoes.tar.gz)

### Code Details
Code is pretty self explanatory. There are mainly four files in implementation:

* main.py:  To train and load the model
* utils.py:   Contains the helper function to download, preprocess the dataset and also to define Generators and Discriminator networks
* parameters.py: Contains the parameter for the model
* DiscoGAN.py -- Contains the DiscoGAN class defining the model



