# Extra unused APIs for uploading/deleting images on server

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