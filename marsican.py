import os
from flask import Flask, flash, request, redirect, url_for, send_from_directory
from werkzeug.utils import secure_filename
from os.path import join, dirname, realpath

app = Flask(__name__)

@app.route('/')
def hello_world():
    return 'Hello, World!'
