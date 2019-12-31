from flask import Flask
from flask import jsonify, make_response
from flask import request, redirect, abort

import os
import urllib.request
from werkzeug.utils import secure_filename

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

# Helper methods
def allowed_filename(filename):
    if '.' not in filename:
        return False
    return filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def setup_model():
    data_dir = "../../data/"
    weights_path = data_dir + "saved_models/best_xception_model.hdf5"


# Setup backend model
x_model = XceptionModel()


# REST API methods
@app.errorhandler(404)
def not_found(error):
    return make_response(jsonify({'error': 'Not found'}), 404)


@app.route('/')
def index():
    return "Hello, World!\n"


@app.route('/status', methods=['GET'])
def status():
    return "Running...\n"


@app.route('/upload/image', methods=['POST'])
def upload_img():
    # check if post request has files
    if 'file' not in request.files:
        resp = jsonify({'error' : 'No file part in the request'})
        resp.status_code = 400
        return resp

    file = request.files['file']
    
    # check if file selected
    if file.filename == '':
	    resp = jsonify({'error' : 'No file selected for uploading'})
	    resp.status_code = 400
	    return resp

    if not allowed_filename(file.filename):
        resp = jsonify({'error' : 'file extension must be (.png, .jpg, .jpeg)'})
        resp.status_code = 400
        return resp

    # valid file uploaded
    if file:
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        resp = jsonify({'message' : 'File successfully uploaded'})
        resp.status_code = 201
        return resp


@app.route('/predict', methods=['POST'])
def predict_img():
    # TODO: Implement
    return x_model.status()


if __name__ == "__main__":
    app.run(port=5001)