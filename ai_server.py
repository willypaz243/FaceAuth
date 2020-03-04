import os

import cv2
import dlib
import numpy as np
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename

from tensorflow.keras.models import load_model

from face_auth.FaceDetector import FaceDetector
from face_auth.ImgProcessor import ImgProcessor
from face_auth.FaceEncoder import FaceEncoder
from face_auth.face_model import face_model
from face_auth.kmean import K_mean
from face_auth.Identifier import Identifier

predictor = dlib.get_frontal_face_detector()
face_detector = FaceDetector(predictor)

landmark_detector = dlib.shape_predictor("face_auth/datas/shape_predictor_68_face_landmarks.dat")
img_processor = ImgProcessor(landmark_detector)

model = load_model("face_auth/models/facenet_keras.h5")
face_encoder = FaceEncoder(model)

k_mean = K_mean(model_name="testing_web")

identifier = Identifier(face_encoder, k_mean, "test_identifier_web")

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = './images'
app.config['TEMPLATES_FOLDER'] = './templates'

def get_images(uploaded_files):
    """
    Obtiene la matriz de varios archivos JPEG.
    
    Args:
    
        - uploaded_files: los archivos que se subieron al server.
    
    Returns:
    
        Una `Numpy.array` matriz de imagenes.
    """
    images = []  # ['image_field']
    for archivo in uploaded_files:
        filename = secure_filename(archivo.filename)
        if not os.path.exists(app.config['UPLOAD_FOLDER']):
            os.system('mkdir '+app.config['UPLOAD_FOLDER'])
        uploaded_file_url = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        archivo.save(uploaded_file_url)
        #print(filename)
        img = cv2.imread(uploaded_file_url)
        images.append(img)
    return images


@app.route('/registro', methods=['GET', 'POST'])
def upload_image():
    """
    Sube multiples archivos JPG para ser evaluados y registrados por el modelo de identificación.
    """
    print(request.method)
    if request.method == 'POST':
        _id = int(request.form.get("int_field"))
        uploaded_files = request.files.getlist("image_field")
        # Obtiene las imagenes subidas
        images = get_images(uploaded_files)
        # Obtiene imagenes con el rostro recortado.
        face_images = get_faces_images(images)
        # Registra las imagenes
        registrado = identifier.register(_id, face_images)

        os.system('rm -r '+os.path.join(app.config['UPLOAD_FOLDER']) + '/*') # para no dejar imagenes guardadas.

        data_user = {"registrado":registrado}
        return jsonify(data_user)
    else:
        return render_template('register_face.html')

@app.route('/identify', methods=['GET', 'POST'])
def identify_images():
    """
    Evalua una serie de imagenes para identificar la identidad de un usuario.
    """
    print(request.method)
    if request.method == 'POST':
        uploaded_files = request.files.getlist("image_field")
        images = get_images(uploaded_files)
        
        face_images = get_faces_images(images)
        
        _id = identifier.identify(face_images)

        os.system('rm -r '+os.path.join(app.config['UPLOAD_FOLDER']) + '/*')

        data_user = {"id":_id}
        return jsonify(data_user)
    else:
        return render_template("identify_face.html")

def get_faces_images(images):
    face_images = []
    for i, image in enumerate(images):
    
        faces = face_detector(image)
        
        if len(faces) == 1:
            face = faces[0]
            margin = np.round(face * 0.15).astype('int') * [[-1],[1]]
            face += margin
            (x1, y1),(x2, y2) = face
            if face.min() >= 0:
                face_img = image[y1: y2, x1: x2]
                face_img = img_processor(face_img, margin)
                face_images.append(face_img)
    return face_images