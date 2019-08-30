from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
from tensorflow import keras

from tensorflow.python.ops import array_ops

from .inception_blocks2 import inception_block_1a, inception_block_1b, inception_block_1c,\
    inception_block_2a, inception_block_2b, inception_block_3a, inception_block_3b

#def convolution_layer(filter, size)

def face_model():

    input_layer = tf.keras.layers.Input(shape=[96,96,3])

    initializer = tf.random_normal_initializer(0., 0.02)

    x = keras.layers.ZeroPadding2D((3, 3))(input_layer)

    x = keras.layers.Conv2D(64, 7, strides=2, kernel_initializer=initializer, name='conv1')(x)
    x = keras.layers.BatchNormalization(name = 'bn1')(x)
    x = keras.layers.Activation('relu')(x)

    x = keras.layers.ZeroPadding2D((1, 1))(x)
    x = keras.layers.MaxPooling2D((3, 3), strides = 2)(x)
    
    x = keras.layers.Conv2D(64, 1, strides=1, kernel_initializer=initializer, name='conv2')(x)
    x = keras.layers.BatchNormalization(name='bn2')(x)
    x = keras.layers.Activation('relu')(x)

    x = keras.layers.ZeroPadding2D((1, 1))(x)

    x = keras.layers.Conv2D(192, 3, strides=1, kernel_initializer=initializer, name='conv3')(x)
    x = keras.layers.BatchNormalization(name = 'bn3')(x)
    x = keras.layers.Activation('relu')(x)

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

    return keras.Model(inputs=input_layer, outputs=x, name='FaceRecoModel')


"""

modelo = face_model()
image = tf.io.read_file('media/images/Jdk_2.jpg')
image = tf.image.decode_jpeg(image)
image = tf.image.resize(image, (96,96))
#image = array_ops.transpose(image)
image = image / 255
input_image = tf.cast([image], tf.float32)
result = modelo.predict(input_image)
print(type(result))
"""