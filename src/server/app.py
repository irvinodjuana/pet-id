from flask import Flask
from flask import jsonify, make_response
from flask import request, redirect, abort

import os
import urllib.request
from werkzeug.utils import secure_filename

from keras.applications.xception import Xception

import sys
sys.path.append('../')

from model.xception_model import XceptionModel

# File upload config
dirname = os.path.dirname(__file__)
UPLOAD_FOLDER = os.path.join(dirname, 'uploads')
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])

# App config
app = Flask(__name__)
app.secret_key = 'dev'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# Backend model
model = None

# Helper methods
def allowed_filename(filename):
    """Check if filename is acceptable to upload"""
    if '.' not in filename:
        return False
    return filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def setup_model():
    """Setup backend machine learning model for prediction"""
    global model

    data_dir = "../../data/"
    weights_path = data_dir + "saved_models/best_xception_model.hdf5"
    breeds_path = data_dir + "assets/dog_breeds.txt"

    model = XceptionModel()
    model.load_pretrained_model(weights_path, breeds_path)
    print("Model setup complete.\n")


# REST API methods
@app.errorhandler(404)
def not_found(error):
    return make_response(jsonify({'error': 'Not found'}), 404)


@app.route('/')
def index():
    """Base URI"""
    # TODO: redirect to webpage?
    return "Hello, World!\n"


@app.route('/status', methods=['GET'])
def status():
    """Status check for server"""
    return "Running...\n"


@app.route('/predict', methods=['POST'])
def predict():
    """Return a prediction based on imagefile sent by POST request"""
    # Check image uploaded properly to endpoint
    if request.method == 'POST':
        if request.files.get('image'):
            img = request.files['image'].read()

            # Check if number of breeds specified
            try:
                num_results = int(request.form['num_results'])
            except:
                num_results = 5
            # Fetch breed predictions and dog detection
            predictions = model.predict_img(img, num_results)
            dog_found = model.detect_dog_img(img)

            resp = jsonify({'predictions': predictions, 'dog_found': dog_found})
            resp.status_code = 201
            return resp

    resp = jsonify({'error' : "image not found"})
    resp.status_code = 400
    return resp

# APIs for uploading/deleting images on server

# @app.route('/image/upload', methods=['POST'])
# def upload_img():
#     """Uploads an image for temporary storage on server"""
#     # check if post request has files
#     if 'file' not in request.files:
#         resp = jsonify({'error' : 'No file part in the request'})
#         resp.status_code = 400
#         return resp

#     file = request.files['file']
    
#     # check if file selected
#     if file.filename == '':
# 	    resp = jsonify({'error' : 'No file selected for uploading'})
# 	    resp.status_code = 400
# 	    return resp

#     if not allowed_filename(file.filename):
#         resp = jsonify({'error' : 'file extension must be (.png, .jpg, .jpeg)'})
#         resp.status_code = 400
#         return resp

#     # valid file uploaded
#     if file:
#         filename = secure_filename(file.filename)
#         file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
#         resp = jsonify({'message' : 'File successfully uploaded'})
#         resp.status_code = 201
#         return resp


# @app.route('/image/<filename>', methods=['DELETE'])
# def delete_img(filename):
#     """Deletes an image from storage"""
#     filename = secure_filename(filename)
#     filename = os.path.join(app.config['UPLOAD_FOLDER'], filename)
#     # Check if file is an image in the uploads folder
#     # Delete if correct
#     if allowed_filename(filename) and os.path.exists(filename):
#         os.remove(filename)
#         resp = jsonify({'message' : 'File successfully deleted'})
#         resp.status_code = 204
#     else:
#         resp = jsonify({'error' : 'file not found on server'})
#         resp.status_code = 404
    
#     return resp

# @app.route('/predict/file', methods=['POST'])
# def predict_file():
#     """Returns a prediction for an image stored"""
#     # Find filename in request
#     if not request.json or not 'filename' in request.json:
#         resp = jsonify({'error' : 'filename not found in request'})
#         resp.status_code = 400
#         return resp

#     filename = request.json['filename']
#     filename = os.path.join(app.config['UPLOAD_FOLDER'], filename)
#     # Check if file exists on server
#     if not os.path.exists(filename):
#         resp = jsonify({'error' : 'file not found on server'})
#         resp.status_code = 404
#         return resp
    
#     # see if number of breed predictions specified
#     if 'num_results' in request.json:
#         num_results = request.json['num_results']
#     else:
#         num_results = 5

#     predictions = model.predict_file(filename, num_results)
#     dog_found = model.detect_dog_file(filename)

#     resp = jsonify({'predictions': predictions, 'dog_found': dog_found})
#     resp.status_code = 201
#     return resp
    

if __name__ == "__main__":
    setup_model()
    app.run(port=5000, threaded=False)