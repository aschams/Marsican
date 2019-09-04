import os
import sys
from flask import Flask, flash, request, redirect, url_for, send_from_directory
from werkzeug.utils import secure_filename
from os.path import join, dirname, realpath


import cv2
import numpy as np
import keras
from keras import backend as K
import tensorflow as tf

sys.path.append('src')
import counting_functions as mcf

UPLOAD_FOLDER = join(dirname(realpath(__file__)), 'img/uploads/')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def load_keras_model():
    """Load in the pre-trained model"""
    global model
    model = keras.models.load_model('Models/balanced_gray_classes.hdf5')
    # Required for model to work
    global graph
    graph = tf.get_default_graph()


def mouse_crop(event, x, y, flags, param):
    # grab references to the global variables
    global x_start, y_start, x_end, y_end, cropping

    # if the left mouse button was DOWN, start RECORDING
    # (x, y) coordinates and indicate that cropping is being
    if event == cv2.EVENT_LBUTTONDOWN:
        x_start, y_start, x_end, y_end = x, y, x, y
        cropping = True

    # Mouse is Moving
    elif event == cv2.EVENT_MOUSEMOVE:
        if cropping == True:
            x_end, y_end = x, y

    # if the left mouse button was released
    elif event == cv2.EVENT_LBUTTONUP:
        # record the ending (x, y) coordinates
        x_end, y_end = x, y
        cropping = False # cropping is finished

        refPoint = [(x_start, y_start), (x_end, y_end)]

        if len(refPoint) == 2: #when two points were found
            roi = oriImage[refPoint[0][1]:refPoint[1][1], refPoint[0][0]:refPoint[1][0]]
            cv2.imshow("Cropped", roi)


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            flash('No selected file''')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            print('\n'*2, filename, '\n'*2)
            file_name = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_name)

            load_keras_model()
            global graph
            graph = tf.get_default_graph()
            with graph.as_default():
                mcf.complete_fit(file_name,
                                 template='img/colonies/MJF465_600mOsm_40_colony.jpg',
                                 res_img1='who_cares.jpg',
                                 res_img2='img/results/' + filename,
                                 model_=model)
            return redirect(url_for('uploaded_file',
                                    filename=filename))
    return '''
    <!doctype html>
    <title>Upload new File</title>
    <h1>Upload new File</h1>
    <form method=post enctype=multipart/form-data>
      <input type=file name=file>
      <input type=submit value=Uploadz>
    </form>
    '''


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory('img/results/', filename)
