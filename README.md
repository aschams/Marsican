# Marsican

Marsican is a neural network based colony counter. It uses template matching to find colonies, and then a neural network to count them. It can be launched locally or be found on [Heroku](marsican.herokuapp.com).

Counting colonies is a common activity in biology, and can be slow, boring, and take up a significant amount of time if a large number of plates must be counted. The goal of Marsican is to automate this counting process, saving time and effort in counting. Because Marsican is a small personal project, I have set the goal of counting plates with up to 50 colonies within 10% accuracy. Experimentally, Marsican is most successful counting plates with spread out, consistently shaped/sized colonies, and can likely count higher than this.

## Sturcture of the Repository

### Models

Contains an hdf5 file containing the model used. It is called balanced_gray_classes.hdf5 because 1. the model was trained using balanced class weights and 2. images are converted to grayscale automatically prior to being fed into the model.

### Notebooks
This project includes 7 notebooks performing different functions.

technical_notebook.ipynb goes through the full process of analysis, from moving the data into the correct folders for the training/validation split to running the model. It works on a small subset of the data, and exists to showcase the pipeline used. 

#### Dirty Notebooks

Creating_Train-validation_split.ipynb	is used to create the training/validation split for training the model. Because only one model is being trained, an additional test set is not necessary.

Data_Transferring.ipynb	contains the code used to build folders that would house the data, prior to training/validation split. 

Detect_Circles.ipynb contains code used to locate the plate (functionality not used in final model), testing small circle detection (functionality dropped in final model) and testing of template matching.

EDA_Cleaning.ipynb contains code used to access and visualize full plate images from [MicroBIA](http://www.microbia.org/index.php/resources)

Model.ipynb	includes all code used to train the model. The final model used is the last model in this notebook. The model is taken from [A. Ferrari, et al., Bacterial colony counting with Convolutional Neural Networks in Digital Microbiology Imaging, Pattern Recognition (2016),](https://www.semanticscholar.org/paper/Bacterial-colony-counting-with-Convolutional-Neural-Ferrari-Lombardi/646cc8ef9bc7b41fb6297c45a092b5628d5da5d0) The only adjustment made was using batch normalization instead of Local Response Normalization because LRN is not natively supported in Keras.

connecting_to_aws.ipynb contains the code used to connect to aws and upload the full-plate images to AWS. These images ended up not being used.

### sample_data

Contains a small amount of data to showcase the full pipeline. blood_agar_imgs contain sample images from MicroBIA, while clear_agar_and_zero_colony_imgs contains images from the Sukharev group. 

labelled_imgs contains images in their labelled folders. The stratified_data folder contains the images from the other folders in labelled_imgs directory and has them split into a training and validation set for use in Keras.

The web_app directory contains sample data that can be directly input into the Marsican web app. The plate folder contains 2 full plate images and the colony folder contains the 2 corresponding template colonies for matching.

### img

Folders used by the Marsican web app.

### src
Functions used to clean the data can be found in marsican_functions.py

Functions used to augment the data prior to fitting the model can be found in augment_functions.py

Functions used to analyze the data can be found in the counting_functions.py file.

### static/templates

Contain the css and html of the web app.

## Data

Data used to train the model was acquired from [MicroBIA](http://www.microbia.org/index.php/resources) (The Segments Enumeration Dataset) and the Sukharev Group at the University of Maryland, College Park.
