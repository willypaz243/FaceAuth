import time
from multiprocessing.dummy import Pool
import glob
from inception_blocks import *

# Especifica que el modelo recibira la entrada 
# de la forma por ejemplo (3, 96, 96)
# y no de la forma        (96, 96, 3)
K.set_image_data_format('channels_first')



def faceRecoModel(input_shape = (3, 96, 96)):
    """
    Implementación del modelo Inception utilizado para FaceNet.
    Parametros:
    input_shape, Forma de las imágenes del conjunto de datos.
    """
    # Define la entrada como tensor con forma input_shape
    input_layer = keras.layers.Input(input_shape)

    # Zero-Padding
    hidden_layer = keras.layers.ZeroPadding2D((3, 3))(input_layer)

    # Primer bloque
    hidden_layer = keras.layers.Conv2D(64, (7, 7), strides = (2, 2), name = 'conv1')(hidden_layer)
    hidden_layer = keras.layers.normalization.BatchNormalization(axis = 1, name = 'bn1')(hidden_layer)
    hidden_layer = keras.layers.Activation('relu')(hidden_layer)

    # Zero-Padding + MAXPOOL
    hidden_layer = keras.layers.ZeroPadding2D((1, 1))(hidden_layer)
    hidden_layer = keras.layers.pooling.MaxPooling2D((3, 3), strides = 2)(hidden_layer)

    # Segundo bloque
    hidden_layer = keras.layers.Conv2D(64, (1, 1), strides = (1, 1), name = 'conv2')(hidden_layer)
    hidden_layer = keras.layers.normalization.BatchNormalization(axis = 1, epsilon=0.00001, name = 'bn2')(hidden_layer)
    hidden_layer = keras.layers.Activation('relu')(hidden_layer)

    # Zero-Padding + MAXPOOL
    hidden_layer = keras.layers.ZeroPadding2D((1, 1))(hidden_layer)

    # Tercer bloque
    hidden_layer = keras.layers.Conv2D(192, (3, 3), strides = (1, 1), name = 'conv3')(hidden_layer)
    hidden_layer = keras.layers.normalization.BatchNormalization(axis = 1, epsilon=0.00001, name = 'bn3')(hidden_layer)
    hidden_layer = keras.layers.Activation('relu')(hidden_layer)

    # Zero-Padding + MAXPOOL
    hidden_layer = keras.layers.ZeroPadding2D((1, 1))(hidden_layer)
    hidden_layer = keras.layers.pooling.MaxPooling2D(pool_size = 3, strides = 2)(hidden_layer)

    # Inception 1: a/b/c
    hidden_layer = inception_block_1a(hidden_layer)
    hidden_layer = inception_block_1b(hidden_layer)
    hidden_layer = inception_block_1c(hidden_layer)

    # Inception 2: a/b
    hidden_layer = inception_block_2a(hidden_layer)
    hidden_layer = inception_block_2b(hidden_layer)

    # Inception 3: a/b
    hidden_layer = inception_block_3a(hidden_layer)
    hidden_layer = inception_block_3b(hidden_layer)

    # La capa superior
    hidden_layer = keras.layers.pooling.AveragePooling2D(pool_size=(3, 3), strides=(1, 1), data_format='channels_first')(hidden_layer)
    hidden_layer = keras.layers.core.Flatten()(hidden_layer)
    hidden_layer = keras.layers.core.Dense(128, name='dense_layer')(hidden_layer)

    # L2 normalization
    hidden_layer = keras.layers.core.Lambda(lambda  x: K.l2_normalize(x,axis=1))(hidden_layer)

    # Crea el modelo
    model = keras.models.Model(inputs = input_layer, outputs = hidden_layer, name='FaceRecoModel')

    return model