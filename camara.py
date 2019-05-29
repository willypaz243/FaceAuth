import pandas as pd
import cv2 as cv2
import keras
import numpy as np
from id_model import faceRecoModel
from triplet_loss import triplet_loss
from utils import load_weights_from_FaceNet, \
    load_database, get_img_code, get_most_similar,\
        get_face, create_database, load_centroides, load_data
from k_mean import K_mean
import os

# En este caso estamos cargando el modelo
print('Cargando modelo de reconicimiento facial')
model = keras.models.load_model('my_model.h5')

# Fuente del texto para la imagen
font = cv2.FONT_HERSHEY_SIMPLEX

# Copila en modelo
model.compile(optimizer = 'adam', loss = triplet_loss, metrics = ['accuracy'])

# Modelo k-mean
model_k = K_mean()

centroides = load_centroides()
if centroides:
    model_k.set_centroides(np.array(list(centroides.values())))
    model_k.set_nombre(list(centroides.keys()))
dataset = load_data()
if dataset.size > 0:
    model_k.train(dataset)
# Abrimos una captura de opencv
CAP = cv2.VideoCapture()

# Cargamos Clasificador en cascada que reconocera rostros humanos.
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

print('LISTO!!')

def camara(ip = None):
    """
    Implementacion de una camara web para identificar 
    personas con sus rostros.

    Puede usar una camara wifi o por IP descomentando las siguiente linea
    Reemplace el IP de muestra con el IP de su camara.

    """
    if ip == None:
        active = CAP.open(0)
    else:
        active = CAP.open(ip+'/video')
    while active:
        _, frame = CAP.read()
        partes, x1, x2, y1, y2 = get_face(frame)
        for parte in partes:
            # Identifica a la persona y estrae su nombre de la foto
            identidad = get_most_similar(get_img_code(parte, model)[0], model_k)
            frame = cv2.putText(frame, identidad, (x1,y1+20), font, 1,(0,0,255),2,cv2.LINE_AA)
            cv2.imshow('parte', parte)
        
        frame = cv2.rectangle(frame,(x1, y1),(x2, y2),(255,0,0),2)
        cv2.imshow('Camara', frame)
        key = cv2.waitKey(1)
        if key == 27: # Preciona Esc para Salir
            break
    CAP.release()
    cv2.destroyAllWindows()

def register_camera(nombre, ip = None):
    """
    Registra y almacena imagenes del rostro del usuario
    y devuelve la minima diferencia que debe tener para la identificación
        nombre, el nombre del usuario 
        ip, si se usa una camara inalambria especifique de forma:
            "http://170.0.0.1:9000/video". Por ejemplo.
    """
    if ip == None:
        active = CAP.open(0)
    else:
        active = CAP.open(ip+'/video')
    while active:
        _, frame = CAP.read()
        partes, x1, x2, y1, y2 = get_face(frame)
        frame = cv2.rectangle(frame,(x1, y1),(x2, y2),(255,0,0),2)
        cv2.imshow('Camara', frame)
        key = cv2.waitKey(1)
        if key == 27: # Preciona Esc para Salir
            dataset = []
            for i in range(20):
                _, frame = CAP.read()
                partes, x1, x2, y1, y2 = get_face(frame)
                parte = partes[0]
                code = get_img_code(parte, model)
                dataset.append(code[0])
            
            dataset = np.array(dataset)
            centroides = load_centroides()
            centroides[nombre] = dataset.mean(axis=0)
            create_database(dataset) # Guarda los datos de las nuevas imagenes
            model_k.set_centroides(np.array(list(centroides.values())))
            model_k.set_nombre(list(centroides.keys()))
            dataframe = pd.DataFrame(centroides)
            if not os.path.exists("centroides"):
                os.mkdir("centroides")
            dataframe.to_csv("centroides/centros.csv")
            break
    CAP.release()
    cv2.destroyAllWindows()
            
        