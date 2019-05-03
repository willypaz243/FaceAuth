import keras
from numpy import genfromtxt
import numpy as np
import os
import cv2
import glob

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
    # Redimenciona a imagen
    img = cv2.resize(img, (96, 96))
    # Cambia el formato de color de BGR a RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # Escala los valores de la imagen en un rango de 0 a 1
    img = np.around(np.transpose(img, (2,0,1))/255.0, decimals=12)
    # Agregamos la imagen a un array como dato de prediccion
    x_train = np.array([img])
    # Devuelve la prediccion hecha por el modelo
    return model.predict_on_batch(x_train)

def get_most_similar(img_input, database):
    """
    Devuelve el nombre de la imagen mas parecida de la base de datos
    a la imagen de entrada,
        img_input, imagen de entrada.
        database, un diccionario de imagenes 
    """
    dist_min = 99999.0
    identidad = None
    for name, img_code in database.items():
        dist = np.linalg.norm(img_code - img_input)
        if dist < dist_min:
            dist_min = dist
            identidad = name
    # Devuelve en nombre de la imagen con menor diferencia 
    # a la imagen de entradas
    return str(identidad)
