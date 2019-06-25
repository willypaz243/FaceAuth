import keras
from numpy import genfromtxt
import numpy as np
import os
import cv2 as cv2
import glob
import pandas as pd

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
EYE_CASCADE = cv2.CascadeClassifier('haarcascade_eye.xml')
# contiene un arreglo con todos los nombres de las capas
# creadas en el modelo y de inception_blocks
WEIGHTS = [
  'conv1', 'bn1', 'conv2', 'bn2', 'conv3', 'bn3',
  'inception_3a_1x1_conv', 'inception_3a_1x1_bn',
  'inception_3a_pool_conv', 'inception_3a_pool_bn',
  'inception_3a_5x5_conv1', 'inception_3a_5x5_conv2', 'inception_3a_5x5_bn1', 'inception_3a_5x5_bn2',
  'inception_3a_3x3_conv1', 'inception_3a_3x3_conv2', 'inception_3a_3x3_bn1', 'inception_3a_3x3_bn2',
  'inception_3b_3x3_conv1', 'inception_3b_3x3_conv2', 'inception_3b_3x3_bn1', 'inception_3b_3x3_bn2',
  'inception_3b_5x5_conv1', 'inception_3b_5x5_conv2', 'inception_3b_5x5_bn1', 'inception_3b_5x5_bn2',
  'inception_3b_pool_conv', 'inception_3b_pool_bn',
  'inception_3b_1x1_conv', 'inception_3b_1x1_bn',
  'inception_3c_3x3_conv1', 'inception_3c_3x3_conv2', 'inception_3c_3x3_bn1', 'inception_3c_3x3_bn2',
  'inception_3c_5x5_conv1', 'inception_3c_5x5_conv2', 'inception_3c_5x5_bn1', 'inception_3c_5x5_bn2',
  'inception_4a_3x3_conv1', 'inception_4a_3x3_conv2', 'inception_4a_3x3_bn1', 'inception_4a_3x3_bn2',
  'inception_4a_5x5_conv1', 'inception_4a_5x5_conv2', 'inception_4a_5x5_bn1', 'inception_4a_5x5_bn2',
  'inception_4a_pool_conv', 'inception_4a_pool_bn',
  'inception_4a_1x1_conv', 'inception_4a_1x1_bn',
  'inception_4e_3x3_conv1', 'inception_4e_3x3_conv2', 'inception_4e_3x3_bn1', 'inception_4e_3x3_bn2',
  'inception_4e_5x5_conv1', 'inception_4e_5x5_conv2', 'inception_4e_5x5_bn1', 'inception_4e_5x5_bn2',
  'inception_5a_3x3_conv1', 'inception_5a_3x3_conv2', 'inception_5a_3x3_bn1', 'inception_5a_3x3_bn2',
  'inception_5a_pool_conv', 'inception_5a_pool_bn',
  'inception_5a_1x1_conv', 'inception_5a_1x1_bn',
  'inception_5b_3x3_conv1', 'inception_5b_3x3_conv2', 'inception_5b_3x3_bn1', 'inception_5b_3x3_bn2',
  'inception_5b_pool_conv', 'inception_5b_pool_bn',
  'inception_5b_1x1_conv', 'inception_5b_1x1_bn',
  'dense_layer'
]

# Un diccionario de las dimenciones de salida (outputs) de cada capa (layer)
conv_shape = {
  'conv1': [64, 3, 7, 7],
  'conv2': [64, 64, 1, 1],
  'conv3': [192, 64, 3, 3],
  'inception_3a_1x1_conv': [64, 192, 1, 1],
  'inception_3a_pool_conv': [32, 192, 1, 1],
  'inception_3a_5x5_conv1': [16, 192, 1, 1],
  'inception_3a_5x5_conv2': [32, 16, 5, 5],
  'inception_3a_3x3_conv1': [96, 192, 1, 1],
  'inception_3a_3x3_conv2': [128, 96, 3, 3],
  'inception_3b_3x3_conv1': [96, 256, 1, 1],
  'inception_3b_3x3_conv2': [128, 96, 3, 3],
  'inception_3b_5x5_conv1': [32, 256, 1, 1],
  'inception_3b_5x5_conv2': [64, 32, 5, 5],
  'inception_3b_pool_conv': [64, 256, 1, 1],
  'inception_3b_1x1_conv': [64, 256, 1, 1],
  'inception_3c_3x3_conv1': [128, 320, 1, 1],
  'inception_3c_3x3_conv2': [256, 128, 3, 3],
  'inception_3c_5x5_conv1': [32, 320, 1, 1],
  'inception_3c_5x5_conv2': [64, 32, 5, 5],
  'inception_4a_3x3_conv1': [96, 640, 1, 1],
  'inception_4a_3x3_conv2': [192, 96, 3, 3],
  'inception_4a_5x5_conv1': [32, 640, 1, 1,],
  'inception_4a_5x5_conv2': [64, 32, 5, 5],
  'inception_4a_pool_conv': [128, 640, 1, 1],
  'inception_4a_1x1_conv': [256, 640, 1, 1],
  'inception_4e_3x3_conv1': [160, 640, 1, 1],
  'inception_4e_3x3_conv2': [256, 160, 3, 3],
  'inception_4e_5x5_conv1': [64, 640, 1, 1],
  'inception_4e_5x5_conv2': [128, 64, 5, 5],
  'inception_5a_3x3_conv1': [96, 1024, 1, 1],
  'inception_5a_3x3_conv2': [384, 96, 3, 3],
  'inception_5a_pool_conv': [96, 1024, 1, 1],
  'inception_5a_1x1_conv': [256, 1024, 1, 1],
  'inception_5b_3x3_conv1': [96, 736, 1, 1],
  'inception_5b_3x3_conv2': [384, 96, 3, 3],
  'inception_5b_pool_conv': [96, 736, 1, 1],
  'inception_5b_1x1_conv': [256, 736, 1, 1],
}




def conv2d_bn(
        layer,
        layer_name=None,
        cv1_out=None,
        cv1_filter=(1, 1),
        cv1_strides=(1, 1),
        cv2_out=None,
        cv2_filter=(3, 3),
        cv2_strides=(1, 1),
        padding=None
    ):
    """
    Crea un tensor de capas,
        layer, la capa inicial
        layer_name, nombre de la capa
        cv1_out, cv2_out, numero de filtros de salida
        cv1_filter, cv2_filter, tamaño del kernel
        cv1_strides, cv2_strides, strides
    """
    num = '' if cv2_out == None else '1'

    tensor = keras.layers.Conv2D(cv1_out, cv1_filter, strides=cv1_strides, data_format='channels_first', name=layer_name+'_conv'+num)(layer)
    tensor = keras.layers.normalization.BatchNormalization(axis=1, epsilon=0.00001, name=layer_name+'_bn'+num)(tensor)
    tensor = keras.layers.Activation('relu')(tensor)

    if padding == None:
        return tensor
    tensor = keras.layers.ZeroPadding2D(padding=padding, data_format='channels_first')(tensor)
    if cv2_out == None:
        return tensor

    tensor = keras.layers.Conv2D(cv2_out, cv2_filter, strides=cv2_strides, data_format='channels_first', name=layer_name+'_conv'+'2')(tensor)
    tensor = keras.layers.normalization.BatchNormalization(axis=1, epsilon=0.00001, name=layer_name+'_bn'+'2')(tensor)
    tensor = keras.layers.Activation('relu')(tensor)
    
    return tensor


def load_weights_from_FaceNet(model):
    """
    Cargar pesos desde archivos csv 
    (que se exportaron desde el modelo de antorcha Openface)
    """
    weights = WEIGHTS
    weights_dict = load_weights()

    # Establecer los pesos de capa del modelo.
    for name in weights:
        if model.get_layer(name) != None:
            model.get_layer(name).set_weights(weights_dict[name])
        elif model.get_layer(name) != None:
            model.get_layer(name).set_weights(weights_dict[name])

def load_weights():
    # Establecer la ruta de los pesos
    dirPath = './weights'
    fileNames = filter(lambda f: not f.startswith('.'), os.listdir(dirPath))
    paths = {}
    weights_dict = {}

    for n in fileNames:
        paths[n.replace('.csv', '')] = dirPath + '/' + n

    for name in WEIGHTS:
        if 'conv' in name:
            conv_w = genfromtxt(paths[name + '_w'], delimiter=',', dtype=None)
            conv_w = np.reshape(conv_w, conv_shape[name])
            conv_w = np.transpose(conv_w, (2, 3, 1, 0))
            conv_b = genfromtxt(paths[name + '_b'], delimiter=',', dtype=None)
            weights_dict[name] = [conv_w, conv_b]
        elif 'bn' in name:
            bn_w = genfromtxt(paths[name + '_w'], delimiter=',', dtype=None)
            bn_b = genfromtxt(paths[name + '_b'], delimiter=',', dtype=None)
            bn_m = genfromtxt(paths[name + '_m'], delimiter=',', dtype=None)
            bn_v = genfromtxt(paths[name + '_v'], delimiter=',', dtype=None)
            weights_dict[name] = [bn_w, bn_b, bn_m, bn_v]
        elif 'dense' in name:
            dense_w = genfromtxt(dirPath+'/dense_w.csv', delimiter=',', dtype=None)
            dense_w = np.reshape(dense_w, (128, 736))
            dense_w = np.transpose(dense_w, (1, 0))
            dense_b = genfromtxt(dirPath+'/dense_b.csv', delimiter=',', dtype=None)
            weights_dict[name] = [dense_w, dense_b]

    return weights_dict


def load_database(model):
    """
    Crea un dicionario con los nombres de las imagenes almacenadas
    como llaves y un codigo proporcionado por el modelo como el dato
    """
    database = {}
    for archivo in glob.glob("images/*"):
        nombre = os.path.splitext(os.path.basename(archivo))[0]
        img = cv2.imread(archivo, 1)
        # Se procede a generar el codigo de la imagen
        database[nombre] = get_img_code(img, model)
    return database

def get_img_code(img, model):
    """
    Genera un codigo a una imagen:
        img, la imagen cargada como una matris.
        model, el modelo que generara su codigo.
    """
    img[:,:,0] = cv2.equalizeHist(img[:,:,0])
    img[:,:,1] = cv2.equalizeHist(img[:,:,1])
    img[:,:,2] = cv2.equalizeHist(img[:,:,2])
    # Redimenciona a imagen
    img = cv2.resize(img, (96, 96))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # Cambia el formato de color de BGR a RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #img = cv2.GaussianBlur(img,(3,3),1)
    img = 255 - img
    # Escala los valores de la imagen en un rango de 0 a 1
    img = np.around(np.transpose(img, (2,0,1))/255.0, decimals=12)
    # Agregamos la imagen a un array como dato de prediccion
    x_train = np.array([img])
    # Devuelve la prediccion hecha por el modelo
    return model.predict_on_batch(x_train)

def get_most_similar(img_input, model_k):
    """
    Devuelve el nombre de la imagen mas parecida de la base de datos
    a la imagen de entrada,
        img_input, imagen de entrada.
        database, un diccionario de imagenes 
    """
    identidad = model_k.that_class(np.array([img_input]))
    return str(identidad)

def get_face(frame):
    """
    Obtiene cuadros de una imagen en la que se encuentren rostros
    Tambien se obtiene cuatro valores que marca los limites del cuadro
        freme, la imagen de la que se obtendra los rostros
    """
    # Detecta los limites de un rostro y los almacena en una lista
    faces = face_cascade.detectMultiScale(frame, 1.1, 5)
    partes = []
    x1 = 0
    x2 = 0
    y1 = 0
    y2 = 0
    for (x, y, w, h) in faces:
        # Fijando dimenciones
        x1 = x
        y1 = y
        x2 = x+w
        y2 = y+h

        y2 = y2 + (y2/100)*18
        y2 = int(y2)

        # Extrae el fracmento de la imagen original que posee el rostro
        parte = frame[y1:y2, x1:x2]
        partes.append(parte)
    return partes, x1, x2, y1, y2

def create_database(dataset):
    """
    Almacena el dataset en un archivo 'csv'.
        dataset -> los datos a almacenar.
    """
    if not os.path.exists("images"):
        os.mkdir("images")
    if os.path.exists("database.csv"):
        dataframe = pd.read_csv("images/database.csv", index_col="Unnamed: 0")
        dataset = np.concatenate([dataframe.to_numpy(),np.array(dataset)])
    dataframe = pd.DataFrame(dataset)
    dataframe.to_csv("images/database.csv")

def load_data():
    """
    Carga los datos para el entrenamiento.
    """
    if os.path.exists("images/database.csv"):
        dataframe = pd.read_csv("images/database.csv", index_col="Unnamed: 0")
        return dataframe.to_numpy()
    else:
        return np.array([])

def load_centroides():
    """
    Carga los centroides.
    Los centroidos son puntos que agrupan los datos de las imagenes
    que predice el modelo.
    """
    try:
        centroides = pd.read_csv("centroides/centros.csv", index_col="Unnamed: 0")
        centroides = centroides.to_dict("list")
        return centroides
    except:
        return {}

def prepare_k_means(model):
    centroides = load_centroides()
    if centroides:
        model.set_centroides(np.array(list(centroides.values())))
        model.set_nombre(list(centroides.keys()))
    dataset = load_data()
    if dataset.size > 0:
        model.train(dataset)

def straighten_image(img, eje, puntos):
    cateto = (puntos[1][1]-puntos[0][1])
    dist = ((puntos[1]-puntos[0])**2).sum()**0.5
    angulo = np.degrees(np.math.asin(cateto/dist))
    m = cv2.getRotationMatrix2D(eje,angulo,1)
    img = cv2.warpAffine(img, m, img[:,:,0].shape)
    return img
def get_eyes_centers(img, x, y):
    eyes = EYE_CASCADE.detectMultiScale(
        img,
        scaleFactor=1.2,
        minNeighbors=6,
        minSize=(28,28),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    res = len(eyes) == 2
    centros = []
    if res:
        for i,j,k,l in eyes:
            centros.append([x+(i+k//2),y+(j+l//2)])
        centros = np.array(centros)
    return res, centros