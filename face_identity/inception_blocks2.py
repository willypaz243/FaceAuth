
from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
from tensorflow import keras

# Implementacion de los bloques.

initializer = tf.random_normal_initializer(0., 0.02)

def inception_block_1a(hidden_layer):
    
    # 3x3========================================================
    layer_3x3 = keras.layers.Conv2D(96, (1, 1), kernel_initializer=initializer, name ='inception_3a_3x3_conv1')(hidden_layer)
    layer_3x3 = keras.layers.BatchNormalization(epsilon=0.00001, name = 'inception_3a_3x3_bn1')(layer_3x3)
    layer_3x3 = keras.layers.Activation('relu')(layer_3x3)
    layer_3x3 = keras.layers.ZeroPadding2D(padding=(1, 1))(layer_3x3)
    layer_3x3 = keras.layers.Conv2D(128, (3, 3), kernel_initializer=initializer, name='inception_3a_3x3_conv2')(layer_3x3)
    layer_3x3 = keras.layers.BatchNormalization(epsilon=0.00001, name='inception_3a_3x3_bn2')(layer_3x3)
    layer_3x3 = keras.layers.Activation('relu')(layer_3x3)
    # ===========================================================
    # 5x5========================================================
    layer_5x5 = keras.layers.Conv2D(16, (1, 1), kernel_initializer=initializer, name='inception_3a_5x5_conv1')(hidden_layer)
    layer_5x5 = keras.layers.BatchNormalization(epsilon=0.00001, name='inception_3a_5x5_bn1')(layer_5x5)
    layer_5x5 = keras.layers.Activation('relu')(layer_5x5)
    layer_5x5 = keras.layers.ZeroPadding2D(padding=(2, 2))(layer_5x5)
    layer_5x5 = keras.layers.Conv2D(32, (5, 5), kernel_initializer=initializer, name='inception_3a_5x5_conv2')(layer_5x5)
    layer_5x5 = keras.layers.BatchNormalization(epsilon=0.00001, name='inception_3a_5x5_bn2')(layer_5x5)
    layer_5x5 = keras.layers.Activation('relu')(layer_5x5)
    # ===========================================================
    # Pool=======================================================
    layer_pool = keras.layers.MaxPooling2D(pool_size=3, strides=2)(hidden_layer)
    layer_pool = keras.layers.Conv2D(32, (1, 1), kernel_initializer=initializer, name='inception_3a_pool_conv')(layer_pool)
    layer_pool = keras.layers.BatchNormalization(epsilon=0.00001, name='inception_3a_pool_bn')(layer_pool)
    layer_pool = keras.layers.Activation('relu')(layer_pool)
    layer_pool = keras.layers.ZeroPadding2D(padding=((3, 4), (3, 4)))(layer_pool)
    # ===========================================================
    # 1x1========================================================
    layer_1x1 = keras.layers.Conv2D(64, (1, 1), kernel_initializer=initializer, name='inception_3a_1x1_conv')(hidden_layer)
    layer_1x1 = keras.layers.BatchNormalization(epsilon=0.00001, name='inception_3a_1x1_bn')(layer_1x1)
    layer_1x1 = keras.layers.Activation('relu')(layer_1x1)
    # ===========================================================
    
    # Concatenando los bloques
    inception = keras.layers.concatenate([layer_3x3, layer_5x5, layer_pool, layer_1x1])

    return inception

def inception_block_1b(hidden_layer):
    # 3x3========================================================
    layer_3x3 = keras.layers.Conv2D(96, (1, 1), kernel_initializer=initializer, name='inception_3b_3x3_conv1')(hidden_layer)
    layer_3x3 = keras.layers.BatchNormalization(epsilon=0.00001, name='inception_3b_3x3_bn1')(layer_3x3)
    layer_3x3 = keras.layers.Activation('relu')(layer_3x3)
    layer_3x3 = keras.layers.ZeroPadding2D(padding=(1, 1))(layer_3x3)
    layer_3x3 = keras.layers.Conv2D(128, (3, 3), kernel_initializer=initializer, name='inception_3b_3x3_conv2')(layer_3x3)
    layer_3x3 = keras.layers.BatchNormalization(epsilon=0.00001, name='inception_3b_3x3_bn2')(layer_3x3)
    layer_3x3 = keras.layers.Activation('relu')(layer_3x3)
    # ===========================================================
    # 5x5========================================================
    layer_5x5 = keras.layers.Conv2D(32, (1, 1), kernel_initializer=initializer, name='inception_3b_5x5_conv1')(hidden_layer)
    layer_5x5 = keras.layers.BatchNormalization(epsilon=0.00001, name='inception_3b_5x5_bn1')(layer_5x5)
    layer_5x5 = keras.layers.Activation('relu')(layer_5x5)
    layer_5x5 = keras.layers.ZeroPadding2D(padding=(2, 2))(layer_5x5)
    layer_5x5 = keras.layers.Conv2D(64, (5, 5), kernel_initializer=initializer, name='inception_3b_5x5_conv2')(layer_5x5)
    layer_5x5 = keras.layers.BatchNormalization(epsilon=0.00001, name='inception_3b_5x5_bn2')(layer_5x5)
    layer_5x5 = keras.layers.Activation('relu')(layer_5x5)
    # ===========================================================
    # Pool=======================================================
    layer_pool = keras.layers.AveragePooling2D(pool_size=(3, 3), strides=(3, 3))(hidden_layer)
    layer_pool = keras.layers.Conv2D(64, (1, 1), kernel_initializer=initializer, name='inception_3b_pool_conv')(layer_pool)
    layer_pool = keras.layers.BatchNormalization(epsilon=0.00001, name='inception_3b_pool_bn')(layer_pool)
    layer_pool = keras.layers.Activation('relu')(layer_pool)
    layer_pool = keras.layers.ZeroPadding2D(padding=(4, 4))(layer_pool)
    # ===========================================================
    # 1x1========================================================
    layer_1x1 = keras.layers.Conv2D(64, (1, 1), kernel_initializer=initializer, name='inception_3b_1x1_conv')(hidden_layer)
    layer_1x1 = keras.layers.BatchNormalization(epsilon=0.00001, name='inception_3b_1x1_bn')(layer_1x1)
    layer_1x1 = keras.layers.Activation('relu')(layer_1x1)
    # ===========================================================
    
    inception = keras.layers.concatenate([layer_3x3, layer_5x5, layer_pool, layer_1x1])

    return inception


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

    tensor = keras.layers.Conv2D(cv1_out, cv1_filter, strides=cv1_strides, kernel_initializer=initializer, name=layer_name+'_conv'+num)(layer)
    tensor = keras.layers.BatchNormalization(epsilon=0.00001, name=layer_name+'_bn'+num)(tensor)
    tensor = keras.layers.Activation('relu')(tensor)

    if padding == None:
        return tensor
    tensor = keras.layers.ZeroPadding2D(padding=padding)(tensor)
    if cv2_out == None:
        return tensor

    tensor = keras.layers.Conv2D(cv2_out, cv2_filter, strides=cv2_strides, kernel_initializer=initializer, name=layer_name+'_conv'+'2')(tensor)
    tensor = keras.layers.BatchNormalization(epsilon=0.00001, name=layer_name+'_bn'+'2')(tensor)
    tensor = keras.layers.Activation('relu')(tensor)
    
    return tensor


def inception_block_1c(hidden_layer):
    # Utilizando una funcion de utils.conv2d_bn
    layer_3x3 = conv2d_bn(
        layer=hidden_layer,
        layer_name='inception_3c_3x3',
        cv1_out=128,
        cv1_filter=(1, 1),
        cv2_out=256,
        cv2_filter=(3, 3),
        cv2_strides=(2, 2),
        padding=(1, 1)
    )

    layer_5x5 = conv2d_bn(
        layer=hidden_layer,
        layer_name='inception_3c_5x5',
        cv1_out=32,
        cv1_filter=(1, 1),
        cv2_out=64,
        cv2_filter=(5, 5),
        cv2_strides=(2, 2),
        padding=(2, 2)
    )

    layer_pool = keras.layers.MaxPooling2D(pool_size=3, strides=2)(hidden_layer)
    layer_pool = keras.layers.ZeroPadding2D(padding=((0, 1), (0, 1)))(layer_pool)

    inception = keras.layers.concatenate([layer_3x3, layer_5x5, layer_pool])

    return inception

def inception_block_2a(hidden_layer):
    layer_3x3 = conv2d_bn(
        layer=hidden_layer,
        layer_name='inception_4a_3x3',
        cv1_out=96,
        cv1_filter=(1, 1),
        cv2_out=192,
        cv2_filter=(3, 3),
        cv2_strides=(1, 1),
        padding=(1, 1)
    )
    layer_5x5 = conv2d_bn(
        layer=hidden_layer,
        layer_name='inception_4a_5x5',
        cv1_out=32,
        cv1_filter=(1, 1),
        cv2_out=64,
        cv2_filter=(5, 5),
        cv2_strides=(1, 1),
        padding=(2, 2)
    )

    layer_pool = keras.layers.AveragePooling2D(pool_size=(3, 3), strides=(3, 3))(hidden_layer)
    layer_pool = conv2d_bn(
        layer=layer_pool,
        layer_name='inception_4a_pool',
        cv1_out=128,
        cv1_filter=(1, 1),
        padding=(2, 2)
    )
    layer_1x1 = conv2d_bn(
        layer=hidden_layer,
        layer_name='inception_4a_1x1',
        cv1_out=256,
        cv1_filter=(1, 1)
    )
    inception = keras.layers.concatenate([layer_3x3, layer_5x5, layer_pool, layer_1x1])

    return inception

def inception_block_2b(hidden_layer):
    #inception4e
    layer_3x3 = conv2d_bn(
        layer=hidden_layer,
        layer_name='inception_4e_3x3',
        cv1_out=160,
        cv1_filter=(1, 1),
        cv2_out=256,
        cv2_filter=(3, 3),
        cv2_strides=(2, 2),
        padding=(1, 1)
    )
    layer_5x5 = conv2d_bn(
        layer=hidden_layer,
        layer_name='inception_4e_5x5',
        cv1_out=64,
        cv1_filter=(1, 1),
        cv2_out=128,
        cv2_filter=(5, 5),
        cv2_strides=(2, 2),
        padding=(2, 2)
    )

    layer_pool = keras.layers.MaxPooling2D(pool_size=3, strides=2)(hidden_layer)
    layer_pool = keras.layers.ZeroPadding2D(padding=((0, 1), (0, 1)))(layer_pool)

    inception = keras.layers.concatenate([layer_3x3, layer_5x5, layer_pool])

    return inception

def inception_block_3a(hidden_layer):
    layer_3x3 = conv2d_bn(
        layer=hidden_layer,
        layer_name='inception_5a_3x3',
        cv1_out=96,
        cv1_filter=(1, 1),
        cv2_out=384,
        cv2_filter=(3, 3),
        cv2_strides=(1, 1),
        padding=(1, 1)
    )
    layer_pool = keras.layers.AveragePooling2D(pool_size=(3, 3), strides=(3, 3))(hidden_layer)
    layer_pool = conv2d_bn(
        layer=layer_pool,
        layer_name='inception_5a_pool',
        cv1_out=96,
        cv1_filter=(1, 1),
        padding=(1, 1)
    )
    layer_1x1 = conv2d_bn(
        layer=hidden_layer,
        layer_name='inception_5a_1x1',
        cv1_out=256,
        cv1_filter=(1, 1)
    )

    inception = keras.layers.concatenate([layer_3x3, layer_pool, layer_1x1])

    return inception

def inception_block_3b(hidden_layer):
    layer_3x3 = conv2d_bn(
        layer=hidden_layer,
        layer_name='inception_5b_3x3',
        cv1_out=96,
        cv1_filter=(1, 1),
        cv2_out=384,
        cv2_filter=(3, 3),
        cv2_strides=(1, 1),
        padding=(1, 1)
    )
    layer_pool = keras.layers.MaxPooling2D(pool_size=3, strides=2)(hidden_layer)
    layer_pool = conv2d_bn(
        layer=layer_pool,
        layer_name='inception_5b_pool',
        cv1_out=96,
        cv1_filter=(1, 1)
    )
    layer_pool = keras.layers.ZeroPadding2D(padding=(1, 1))(layer_pool)

    layer_1x1 = conv2d_bn(
        layer=hidden_layer,
        layer_name='inception_5b_1x1',
        cv1_out=256,
        cv1_filter=(1, 1)
    )
    inception = keras.layers.concatenate([layer_3x3, layer_pool, layer_1x1])

    return inception

