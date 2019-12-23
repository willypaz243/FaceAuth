from __future__ import absolute_import, division, print_function, unicode_literals
import os

import dlib
import cv2
import numpy as np
from tensorflow.keras.models import load_model, save_model

from .kmean import K_mean
from .Image_processor import rescale_image, get_landmarks, enderezar_imagen, calcular_angulo
from .utils import load_weights_from_FaceNet
from .face_model import face_model


class Identificador:

    def __init__(self, name = 'identificador'):
        #with GRAPH.as_default():
        self.model_name = name
        self.predictor = dlib.get_frontal_face_detector()
        self.face_net = self.__inicializar_face_net()
        self.k_model = self.__inicializar_k_model()

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
            facenet = load_model("./face_identity/model.h5")
            #facenet.compile(optimizer='adam', loss=triplet_loss, metrics=['accuracy'])
        else:
            print('Cargando modelo desde los archivos CSV...')
            facenet = face_model()
            load_weights_from_FaceNet(facenet)
            #facenet.compile(optimizer='adam', loss=triplet_loss, metrics=['accuracy'])
            save_model(facenet, "./face_identity/model.h5")
        return facenet
    
    def box_and_encode(self, image):
        """
        Codifica los rostros de una imagen en vectores de 128 valores.
        
        Tambien obtiene dos puntos de un rectangulos para localizar los rostros codificados.
        
        Args:
            -image: Una matriz Numpy, una imagen que contenga almenos un rostro.
        Returns:
            -Retorna dos puntos de un rectangulo (x1, y1) esquina inicial, (x2, y2) esquina final.
            -Retorna una matriz Numpy de dimenciones (n, 128) donde n la cantidad de rostros encontrados en la imagen.
        """
        scale, image = rescale_image(image)
        faces = self.predictor(image)
        boxes = []
        face_images = []
        margins = []
        for face in faces:
            face_box = np.array([[ face.left(),  face.top()   ],
                                 [ face.right(), face.bottom()]])
            face_box *= (face_box > 0)
            margin = np.round(face_box * 0.2).astype('int') * [[-1],[1]]
            face_box = face_box + margin 
            (x1, y1), (x2, y2) = face_box * np.int32(face_box > 0)
            
            boxes.append(np.int32(face_box / scale))
            face_images.append(image[y1: y2, x1: x2])
            margins.append(margin)
            
        return boxes, self.__encode(face_images, margins)
    
    def __encode(self, face_images, margins):
        """
        Condifica las imagene en vectores e 128 valores.
        Args:
            -face_images: una lista de imagenes que contienen unicamente un rostro en toda su área.
        Returns:
            -Retorna una matriz Numpy de dimenciones (n, 128) donde n la cantidad de rostros encontrados en la imagen.
        """
        images = []
        for image, margin in zip(face_images, margins):
            
            img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            landmark = get_landmarks(img_gray)
            angle = calcular_angulo([landmark[36], landmark[45]])
            image = enderezar_imagen(image, angle)
            # encode
            face_box = np.array([(0, 0), image.shape[:2]]) - margin
            # 
            (x1, y1), (x2, y2) = face_box
            image = image[y1: y2, x1: x2]
            if image.shape.count(0) > 0:
                print(image.shape)
                break
            image = cv2.resize(image, (96,96))
            image = cv2.GaussianBlur(image, (5, 5), 2)
            cv2.imshow('rostro', image)
            image = image / 255
            images.append(image)
            
        if images:
            return self.face_net(np.float32(images)).numpy()
        else:
            return images
