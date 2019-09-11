# Marsican

Marsican is a neural network based colony counter. It uses template matching to find colonies, and then a neural network to count them. It can be launched locally or be found on [Heroku](marsican.herokuapp.com).

Counting colonies is a common activity in biological research, and can be slow, boring, and take up a significant amount of time if a large number of plates must be counted. The goal of Marsican is to automate this counting process, saving time and effort in counting. Because Marsican is a small personal project, I have set the goal of counting plates with up to 50 colonies within 10% accuracy. Experimentally, Marsican is most successful counting plates with spread out, consistently shaped/sized colonies, and can likely count higher than this under ideal conditions.

## Data

Data used to train the model was acquired from [MicroBIA](http://www.microbia.org/index.php/resources) (The Segments Enumeration Dataset) and the Sukharev Group at the University of Maryland, College Park.

Data from MicroBIA included blood agar images of 1-6 colonies per image. There are also images for outlier colonies, in which there are more than 6 colonies and confluential colonies, in which the number of colonies is not countable by humans. This project makes use of all these images except confluential colonies images. 

A small fraction of the data used can be found in the sample_data folder, which is described below.


## Using Marsican

### Input
Marsican works using two images: the full plate image and a subimage of a single colony from the plate (or another similar looking plate) to act as a template. This single colony can be cropped from the base image using any simple photo editing software or web app. 

### Algorithm
The computer then searches the full plate image for subimages that closely match the colony template. The user provides a number to act as a threshold for how closely a subimage matches the template to be labelled a match. This however leads to a problem, in which the computer finds a match, moves one pixel to the right and finds another match to the template. Marsican therefore will search the full image, find all these heavily overlapping matches, and then attempt to merge boundary boxes that are close to one another. This minimizes the total number as well as the overlap between the different boxes. Each of these boxes are then passed individually through a neural network to count the number of colonies in each box. Marsican then sums up these counts, and returns an annotated image with bounding boxes around found colonies and colony counts for each box. The total colony count is in the title of the returned image.

### Installing Marsican

Marsican can most efficiently be run by cloning this repository, installing the necessary packages in requirements.txt, and running it through the command line. It is important to note that the web app is in the marsican.py file instead of an app.py file. It is therefore necessary to run the command `export FLASK_APP=marsican.py` prior to running the app using `flask run`.

Marsican can also be found on [Heroku](marsican.herokuapp.com), although sometimes this does not work well because it's on a free server and Tensorflow takes up a lot of the allocated memory.

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

connecting_to_aws.ipynb contains the code used to connect to aws and upload the full plate images to AWS. These images ended up not being used on AWS.

### sample_data

Contains a small amount of data to showcase the full pipeline. blood_agar_imgs contain sample images from MicroBIA, while clear_agar_and_zero_colony_imgs contains images from the Sukharev group. 

labelled_imgs contains images in their labelled folders. The stratified_data folder contains the images from the other folders in labelled_imgs directory and has them split into a training and validation set for use in Keras.

The web_app directory contains sample data that can be directly input into the Marsican web app. The plate folder contains 5 full plate images and the colony folder contains the 5 corresponding template colonies for matching.

### img

Folders used by the Marsican web app.

### src
Functions used to clean the data can be found in marsican_functions.py

Functions used to augment the data prior to fitting the model can be found in augment_functions.py

Functions used to analyze the data can be found in the counting_functions.py file.

### static/templates

Contain the css and html of the web app.

## Results

Testing Marsican on the plates/colonies in the sample_data folder, we get mixed results. On 3 of the images, the goal of being within 10% of the true count (from humans) is met. These numbers are met with thresholds between 50 and 60. Marsican fails to meet this mark on 2 of the images, with errors of approximately 20%. Interestingly, Marsican consistently undercounts, often missing small colonies near the edges of the dish or failing to recognize small colonies close to colonies it does recognize. It also struggles with higher number of colonies in a bounding box, likely due to the lack of training data for these high counts on clear agar plates. This could be because the colonies are small or due to shading/glare due to the quality of the photograph. 

Testing on 10 plates from the Sukharev lab, of which 5 of them are the sample_data plates, Marsican has an error rate of roughly 10% per plate.

## Looking Forward

In the future, I would like to:
1. Increase the amount of training data on clear agar plates, specifically with >2 colonies.
2. Allow multiple templates to be included to help identify different shapes/sizes of colonies or multiple species on a plate. 


## Presentations

A brief slideshow for this project can be found [here](https://docs.google.com/presentation/d/1IhJfe2dy0ikVc5Ty9H7sdmsrilNvJNPGlKLypF4O97w/edit?usp=sharing).
