from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
from tensorflow import keras

from tensorflow.python.ops import array_ops

from .inception_blocks2 import inception_block_1a, inception_block_1b, inception_block_1c,\
    inception_block_2a, inception_block_2b, inception_block_3a, inception_block_3b

#def convolution_layer(filter, size)

def face_model():
    """
    Esta arquitectura de red nuronal esta basada en el paper https://arxiv.org/pdf/1503.03832.pdf    
    """

    input_layer = tf.keras.layers.Input((96, 96, 3))

    # Zero-Padding
    x = keras.layers.ZeroPadding2D((3, 3))(input_layer)

    # First Block
    x = keras.layers.Conv2D(64, (7, 7), strides=(2, 2), name='conv1')(x)
    x = keras.layers.BatchNormalization(name = 'bn1')(x)
    x = keras.layers.Activation('relu')(x)

    # Zero-Padding + MAXPOOL
    x = keras.layers.ZeroPadding2D((1, 1))(x)
    x = keras.layers.MaxPooling2D((3, 3), strides = 2)(x)

    # Second Block
    x = keras.layers.Conv2D(64, (1, 1), strides = (1, 1), name = 'conv2')(x)
    x = keras.layers.BatchNormalization(epsilon=0.00001, name = 'bn2')(x)
    x = keras.layers.Activation('relu')(x)

    # Zero-Padding + MAXPOOL
    x = keras.layers.ZeroPadding2D((1, 1))(x)

    # Second Block
    x = keras.layers.Conv2D(192, (3, 3), strides = (1, 1), name = 'conv3')(x)
    x = keras.layers.BatchNormalization(epsilon=0.00001, name = 'bn3')(x)
    x = keras.layers.Activation('relu')(x)

    # Zero-Padding + MAXPOOL
    x = keras.layers.ZeroPadding2D((1, 1))(x)
    x = keras.layers.MaxPooling2D(pool_size = 3, strides = 2)(x)

    x = inception_block_1a(x)
    x = inception_block_1b(x)
    x = inception_block_1c(x)
    
    x = inception_block_2a(x)
    x = inception_block_2b(x)
    
    x = inception_block_3a(x)
    x = inception_block_3b(x)

    x = keras.layers.AveragePooling2D(pool_size=(3, 3), strides=(1, 1))(x)

    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(128, name='dense_layer')(x)
    
    x = keras.layers.Lambda(lambda y: keras.backend.l2_normalize(y, axis=1))(x)
    """
"""
    return keras.Model(inputs=input_layer, outputs=x, name='FaceRecoModel')

if __name__ == "__main__":
    model = face_model()
    model.summary()
    print(model.get_layer('bn1').get_weights()[0].shape)