import os

import cv2
import numpy as np
from flask import Flask, request, jsonify
from werkzeug import secure_filename

from face_identity.Identificador import Identificador

IDENTIFICADOR = Identificador("scesi_auth")

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = './images'

def get_images(uploaded_files):
    images = []  # ['image_field']
    for archivo in uploaded_files:
        filename = secure_filename(archivo.filename)
        if not os.path.exists(app.config['UPLOAD_FOLDER']):
            os.system('mkdir '+app.config['UPLOAD_FOLDER'])
        uploaded_file_url = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        archivo.save(uploaded_file_url)
        print(filename)
        img = cv2.imread(uploaded_file_url)
        images.append(img)
    return np.array(images)


@app.route('/registro', methods=['GET', 'POST'])
def upload_image():
    print(request.method)
    if request.method == 'POST':
        _id = int(request.form.get("int_field"))
        uploaded_files = request.files.getlist("image_field")
        images = get_images(uploaded_files)
        
        registrado = IDENTIFICADOR.registrar_usuario(_id, np.array(images))

        os.system('rm -r '+os.path.join(app.config['UPLOAD_FOLDER']) + '/*')

        data_user = {"registrado":registrado}
        return jsonify(data_user)
    else:
        return 'hello, world!!'

@app.route('/identify', methods=['GET', 'POST'])
def identify_images():
    print(request.method)
    if request.method == 'POST':
        uploaded_files = request.files.getlist("image_field")
        images = get_images(uploaded_files)
        
        _id = IDENTIFICADOR.identify(np.array(images))

        os.system('rm -r '+os.path.join(app.config['UPLOAD_FOLDER']) + '/*')

        data_user = {"id":_id}
        return jsonify(data_user)
    else:
        return 'hello, world!!'

