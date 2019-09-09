# Marsican

Marsican is a neural network based colony counter. It uses template matching to find colonies, and then a neural network to count them. It can be launched locally or be found on [Heroku](marsican.herokuapp.com).

## Sturcture of the Repository

### Models

Contains an hdf5 file containing the model used. It is called balanced_gray_classes.hdf5 because 1. the model was trained using balanced class weights and 2. images are converted to grayscale automatically prior to being fed into the model.

### Notebooks
This project includes 6 notebooks performing different functions.
Creating_Train-validation_split.ipynb	is used to create the training/validation split for training the model. Because only one model is being trained, an additional test set is not necessary.
Data_Transferring.ipynb	contains the code used to build folders that would house the data, prior to training/validation split. 
Detect_Circles.ipynb contains code used to locate the plate (functionality not used in final model), testing small circle detection (functionality dropped in final model) and testing of template matching
EDA_Cleaning.ipynb contains code used to access and visualize full plate images from [MicroBIA](http://www.microbia.org/index.php/resources)
Model.ipynb	includes all code used to train the model. The model used is the final model in this notebook
connecting_to_aws.ipynb contains the code used to connect to aws and upload the full-plate images to AWS. These images ended up not being used.

### img

All three folders contain example images to use with Marsican.

### src
Process used to clean the data can be found in marsican_functions.py
Functions used to analyze the data can be found in the counting_functions.py file.

### static/templates

Contain the css and html of the web app.

## Data

Data used to train the model was acquired from [MicroBIA](http://www.microbia.org/index.php/resources) and the Sukharev Group at the University of Maryland, College Park.
