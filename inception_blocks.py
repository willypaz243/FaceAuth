from numpy import genfromtxt
from keras import backend as K
from utils import conv2d_bn
import keras

def inception_block_1a(hidden_layer):
    """
    Implementation of an inception block
    """
    # 3x3========================================================
    layer_3x3 = keras.layers.Conv2D(96, (1, 1), data_format='channels_first', name ='inception_3a_3x3_conv1')(hidden_layer)
    layer_3x3 = keras.layers.normalization.BatchNormalization(axis=1, epsilon=0.00001, name = 'inception_3a_3x3_bn1')(layer_3x3)
    layer_3x3 = keras.layers.Activation('relu')(layer_3x3)
    layer_3x3 = keras.layers.ZeroPadding2D(padding=(1, 1), data_format='channels_first')(layer_3x3)
    layer_3x3 = keras.layers.Conv2D(128, (3, 3), data_format='channels_first', name='inception_3a_3x3_conv2')(layer_3x3)
    layer_3x3 = keras.layers.normalization.BatchNormalization(axis=1, epsilon=0.00001, name='inception_3a_3x3_bn2')(layer_3x3)
    layer_3x3 = keras.layers.Activation('relu')(layer_3x3)
    # ===========================================================
    # 5x5========================================================
    layer_5x5 = keras.layers.Conv2D(16, (1, 1), data_format='channels_first', name='inception_3a_5x5_conv1')(hidden_layer)
    layer_5x5 = keras.layers.normalization.BatchNormalization(axis=1, epsilon=0.00001, name='inception_3a_5x5_bn1')(layer_5x5)
    layer_5x5 = keras.layers.Activation('relu')(layer_5x5)
    layer_5x5 = keras.layers.ZeroPadding2D(padding=(2, 2), data_format='channels_first')(layer_5x5)
    layer_5x5 = keras.layers.Conv2D(32, (5, 5), data_format='channels_first', name='inception_3a_5x5_conv2')(layer_5x5)
    layer_5x5 = keras.layers.normalization.BatchNormalization(axis=1, epsilon=0.00001, name='inception_3a_5x5_bn2')(layer_5x5)
    layer_5x5 = keras.layers.Activation('relu')(layer_5x5)
    # ===========================================================
    # Pool=======================================================
    layer_pool = keras.layers.pooling.MaxPooling2D(pool_size=3, strides=2, data_format='channels_first')(hidden_layer)
    layer_pool = keras.layers.Conv2D(32, (1, 1), data_format='channels_first', name='inception_3a_pool_conv')(layer_pool)
    layer_pool = keras.layers.normalization.BatchNormalization(axis=1, epsilon=0.00001, name='inception_3a_pool_bn')(layer_pool)
    layer_pool = keras.layers.Activation('relu')(layer_pool)
    layer_pool = keras.layers.ZeroPadding2D(padding=((3, 4), (3, 4)), data_format='channels_first')(layer_pool)
    # ===========================================================
    # 1x1========================================================
    layer_1x1 = keras.layers.Conv2D(64, (1, 1), data_format='channels_first', name='inception_3a_1x1_conv')(hidden_layer)
    layer_1x1 = keras.layers.normalization.BatchNormalization(axis=1, epsilon=0.00001, name='inception_3a_1x1_bn')(layer_1x1)
    layer_1x1 = keras.layers.Activation('relu')(layer_1x1)
    # ===========================================================
    
    # CONCAT
    inception = keras.layers.concatenate([layer_3x3, layer_5x5, layer_pool, layer_1x1], axis=1)

    return inception

def inception_block_1b(hidden_layer):
    # 3x3========================================================
    layer_3x3 = keras.layers.Conv2D(96, (1, 1), data_format='channels_first', name='inception_3b_3x3_conv1')(hidden_layer)
    layer_3x3 = keras.layers.normalization.BatchNormalization(axis=1, epsilon=0.00001, name='inception_3b_3x3_bn1')(layer_3x3)
    layer_3x3 = keras.layers.Activation('relu')(layer_3x3)
    layer_3x3 = keras.layers.ZeroPadding2D(padding=(1, 1), data_format='channels_first')(layer_3x3)
    layer_3x3 = keras.layers.Conv2D(128, (3, 3), data_format='channels_first', name='inception_3b_3x3_conv2')(layer_3x3)
    layer_3x3 = keras.layers.normalization.BatchNormalization(axis=1, epsilon=0.00001, name='inception_3b_3x3_bn2')(layer_3x3)
    layer_3x3 = keras.layers.Activation('relu')(layer_3x3)
    # ===========================================================
    # 5x5========================================================
    layer_5x5 = keras.layers.Conv2D(32, (1, 1), data_format='channels_first', name='inception_3b_5x5_conv1')(hidden_layer)
    layer_5x5 = keras.layers.normalization.BatchNormalization(axis=1, epsilon=0.00001, name='inception_3b_5x5_bn1')(layer_5x5)
    layer_5x5 = keras.layers.Activation('relu')(layer_5x5)
    layer_5x5 = keras.layers.ZeroPadding2D(padding=(2, 2), data_format='channels_first')(layer_5x5)
    layer_5x5 = keras.layers.Conv2D(64, (5, 5), data_format='channels_first', name='inception_3b_5x5_conv2')(layer_5x5)
    layer_5x5 = keras.layers.normalization.BatchNormalization(axis=1, epsilon=0.00001, name='inception_3b_5x5_bn2')(layer_5x5)
    layer_5x5 = keras.layers.Activation('relu')(layer_5x5)
    # ===========================================================
    # Pool=======================================================
    layer_pool = keras.layers.pooling.AveragePooling2D(pool_size=(3, 3), strides=(3, 3), data_format='channels_first')(hidden_layer)
    layer_pool = keras.layers.Conv2D(64, (1, 1), data_format='channels_first', name='inception_3b_pool_conv')(layer_pool)
    layer_pool = keras.layers.normalization.BatchNormalization(axis=1, epsilon=0.00001, name='inception_3b_pool_bn')(layer_pool)
    layer_pool = keras.layers.Activation('relu')(layer_pool)
    layer_pool = keras.layers.ZeroPadding2D(padding=(4, 4), data_format='channels_first')(layer_pool)
    # ===========================================================
    # 1x1========================================================
    layer_1x1 = keras.layers.Conv2D(64, (1, 1), data_format='channels_first', name='inception_3b_1x1_conv')(hidden_layer)
    layer_1x1 = keras.layers.normalization.BatchNormalization(axis=1, epsilon=0.00001, name='inception_3b_1x1_bn')(layer_1x1)
    layer_1x1 = keras.layers.Activation('relu')(layer_1x1)
    # ===========================================================
    
    inception = keras.layers.concatenate([layer_3x3, layer_5x5, layer_pool, layer_1x1], axis=1)

    return inception

def inception_block_1c(hidden_layer):
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

    layer_pool = keras.layers.pooling.MaxPooling2D(pool_size=3, strides=2, data_format='channels_first')(hidden_layer)
    layer_pool = keras.layers.ZeroPadding2D(padding=((0, 1), (0, 1)), data_format='channels_first')(layer_pool)

    inception = keras.layers.concatenate([layer_3x3, layer_5x5, layer_pool], axis=1)

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

    layer_pool = keras.layers.pooling.AveragePooling2D(pool_size=(3, 3), strides=(3, 3), data_format='channels_first')(hidden_layer)
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
    inception = keras.layers.concatenate([layer_3x3, layer_5x5, layer_pool, layer_1x1], axis=1)

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

    layer_pool = keras.layers.pooling.MaxPooling2D(pool_size=3, strides=2, data_format='channels_first')(hidden_layer)
    layer_pool = keras.layers.ZeroPadding2D(padding=((0, 1), (0, 1)), data_format='channels_first')(layer_pool)

    inception = keras.layers.concatenate([layer_3x3, layer_5x5, layer_pool], axis=1)

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
    layer_pool = keras.layers.pooling.AveragePooling2D(pool_size=(3, 3), strides=(3, 3), data_format='channels_first')(hidden_layer)
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

    inception = keras.layers.concatenate([layer_3x3, layer_pool, layer_1x1], axis=1)

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
    layer_pool = keras.layers.pooling.MaxPooling2D(pool_size=3, strides=2, data_format='channels_first')(hidden_layer)
    layer_pool = conv2d_bn(
        layer=layer_pool,
        layer_name='inception_5b_pool',
        cv1_out=96,
        cv1_filter=(1, 1)
    )
    layer_pool = keras.layers.ZeroPadding2D(padding=(1, 1), data_format='channels_first')(layer_pool)

    layer_1x1 = conv2d_bn(
        layer=hidden_layer,
        layer_name='inception_5b_1x1',
        cv1_out=256,
        cv1_filter=(1, 1)
    )
    inception = keras.layers.concatenate([layer_3x3, layer_pool, layer_1x1], axis=1)

    return inception

