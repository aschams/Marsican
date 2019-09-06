import os
import sys
from flask import Flask, flash, request, redirect, url_for, send_from_directory, render_template
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
RESULTS_FOLDER = join(dirname(realpath(__file__)), 'img/results/')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULTS_FOLDER'] = RESULTS_FOLDER
app.debug = True

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def load_keras_model():
    """Load in the pre-trained model"""
    global model
    model = keras.models.load_model('Models/balanced_gray_classes.hdf5')
    # Required for model to work
    global graph
    graph = tf.get_default_graph()




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
            return redirect(url_for('crop_img', filename=filename, file_name=file_name))
    return '''
    <!doctype html>
    <title>Upload Plate Image</title>
    <h1>Upload new File</h1>
    <form method=post enctype=multipart/form-data>
      <input type=file name=file>
      <input type=submit value=Upload>
    </form>
    '''


@app.route('/uploads/<filename>', methods=['GET', 'POST'])
def crop_img(filename):
        file_name = request.args.get('file_name')
        print(file_name)
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
                file_name2 = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                threshold=int(request.form['threshold'])
                load_keras_model()
                global graph
                graph = tf.get_default_graph()
                with graph.as_default():
                    tcc = mcf.complete_fit(file_name,
                                           template=file_name2,
                                           res_img1='who_cares.jpg',
                                           res_img2='img/results/' + filename,
                                           model_=model,
                                           threshold=threshold/100)
                K.clear_session()
                return redirect(url_for('uploaded_file',
                                        filename=filename, tcc=tcc[0]))
        return '''
        <!doctype html>
        <title>Upload new File</title>
        <h1>Upload Template</h1>
        <form method=post enctype=multipart/form-data>
          <input type=file name=file>
          <input type="number" min="1" max="100" name="threshold" value=60/>
          <input type=submit value=Upload>
        </form>
        '''

@app.route('/results/<filename>/<tcc>')
def uploaded_file(filename, tcc):
    print("hello")
    full_filename = os.path.join(RESULTS_FOLDER, filename)
    # return render_template("results.html", plate_image=full_filename, tot_count=tcc)
    return send_from_directory('img/results/', filename, as_attachment=True, attachment_filename=f"{tcc}_{filename}")
    # return send_from_directory('img/results/', filename)
