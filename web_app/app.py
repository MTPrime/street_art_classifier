import flask 
from models import * 
from flask import Flask, redirect, render_template, request, url_for
from PIL import Image, ExifTags

from flask import Flask, render_template, flash, request, redirect, url_for, send_from_directory
import sys
import os
from werkzeug.utils import secure_filename
from PIL import Image, ExifTags
import numpy as np
from gevent.pywsgi import WSGIServer
import pandas as pd

FILEPATH = os.path.realpath(__file__)
ROOTPATH = os.path.split(FILEPATH)[0]
UPLOADPATH = os.path.join(ROOTPATH, 'uploads')

app = Flask(__name__)


def rotate_save(f, file_path):
    try:
        image = Image.open(f)
        for orientation in ExifTags.TAGS.keys():
            if ExifTags.TAGS[orientation] == 'Orientation':
                break
        exif = dict(image._getexif().items())

        if exif[orientation] == 3:
            image = image.rotate(180, expand=True)
        elif exif[orientation] == 6:
            image = image.rotate(270, expand=True)
        elif exif[orientation] == 8:
            image = image.rotate(90, expand=True)
        image.save(file_path)
        image.close()

    except (AttributeError, KeyError, IndexError):
        image.save(file_path)
        image.close()


@app.route('/', methods=['GET'])
def index():
    """Render a simple splash page."""
    return render_template('index.html')
    
# def hello_world():
#     return'''
#         <h2> Welcome to my Webpage.</h2>
#         <form action="/predict" method='POST'>
#             <input type="text" name="user_input" />
#             <input type="submit" />
#         </form>
#         <form action="/" >
#             <input type="submit" value = "Go to the homepage."/>
#         </form>
#     '''

@app.route('/styles',methods=['GET', 'POST'])
def styles():
    return render_template('styles.html')

@app.route('/about',methods=['GET', 'POST'])
def about():
    return render_template('about.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')


@app.route('/predictions', methods=["GET", "POST"])
def upload():
    if request.method == 'POST':
        f = request.files["file"]
        file_path = os.path.join(UPLOADPATH, secure_filename(f.filename))
        f.save(file_path)
        # rotate_save(f, file_path)
        predictions = classify_new_image(file_path, classifier)
        return redirect(url_for('uploaded_file',
                        filename=os.path.split(file_path)[1], predictions=predictions))
    if len(os.listdir(UPLOADPATH)) != 0:
        for file in os.listdir(UPLOADPATH):
            os.remove(os.path.join(UPLOADPATH, file))
    return render_template('predictions.html')


@app.route('/show/<filename><predictions>')
def uploaded_file(filename, predictions):
    return render_template('predictions.html', filename=filename, predictions=predictions)

@app.route('/uploads/<filename>')
def send_file(filename):
    return send_from_directory(UPLOADPATH, filename)

if __name__ == '__main__':
    encoder_model='data/best_encoder_decoder.h5'
    classifier_model='data/5_class_model_best.h5'
    classifier = load_model(classifier_model)
    autoencoder = load_model(encoder_model) 
    app.run(app.run(host='0.0.0.0', port=5000, threaded=True))