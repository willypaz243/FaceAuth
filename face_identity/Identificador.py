from __future__ import absolute_import, division, print_function, unicode_literals
import os

import numpy as np
import tensorflow as tf
from tensorflow import keras

from .kmean import K_mean
from .Image_processor import Image_processor
from .utils import load_weights_from_FaceNet
from .face_model import face_model


class Identificador:

    def __init__(self, name = 'identificador'):
        #with GRAPH.as_default():
        self.model_name = name
        self.face_net = self.__inicializar_face_net()
        self.k_model = self.__inicializar_k_model()
        self.img_processor = Image_processor()

    def __inicializar_k_model(self):
        """
        Inicializa el modelo clasificador que te dira a quien pertenece el rostro.
        """
        k_model = K_mean(self.model_name + '_km')
        #self.prepare_k_means(k_model) # Esta funcion no se esta utilizando, en su lugar se utiliza MongoDB para almacenar los datos
        return k_model

    def __inicializar_face_net(self):
        """
        Inicializa la red neuronal que codificará la imagen para su clasificacion.
        """
        facenet = None
        if os.path.exists('./face_identity/model.h5'):
            print("Cargando modelo existente..")
            #facenet = create_face_model()
            facenet = keras.models.load_model("./face_identity/model.h5")
            #facenet.compile(optimizer='adam', loss=triplet_loss, metrics=['accuracy'])
        else:
            print('Cargando modelo desde los archivos CSV...')
            facenet = face_model()
            load_weights_from_FaceNet(facenet)
            #facenet.compile(optimizer='adam', loss=triplet_loss, metrics=['accuracy'])
            keras.models.save_model(facenet, "./face_identity/model.h5")
        return facenet
    
    def __encode_images(self, images):
        """
        Procesa y codifica una lote de imagenes de un rostro para determinar su identidad.
        
        Args:
            - images: Un lote imagenes en forma de matriz tipo numpy.
        """
        faces = self.img_processor.process_image(images)
        face_codes = []
        if faces.any():
            input_faces = tf.cast(faces, tf.float32)
            face_codes = self.face_net.predict(input_faces)
        return face_codes

    # Publicas
    
    def identify(self, images):
        """
        Identifica a una persona usando un lote de imagenes con su rostro.
        
        Args:
            - images: Un lote imagenes en forma de matriz tipo numpy.
        """
        image_codes = self.__encode_images(images)
        code = np.mean(image_codes, axis=0)
        identity = self.k_model.that_class(code)
        return identity

    def registrar_usuario(self, id_user, images):
        """
        Registra un lote de imagenes con el rostro de quie se quiere identificar asignados a un id.
        
        Args:
            - id_user: Un numero entero que se asigna al usuario a identificar.
            - images: Un lote imagenes en forma de matriz tipo numpy.
        """
        image_codes = self.__encode_images(images)
        registrado = False
        if len(image_codes) > 0:
            registrado = self.k_model.add_class(id_user, image_codes)
        return registrado