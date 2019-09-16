import sys
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras

sys.path.append('..')

from triplet_loss import TripletLossLayer
from face_identity.face_model import face_model
from dataset_loader import dataset
from model import create_model




def fit_model():
    model = create_model()
    
    input_1 = keras.layers.Input( shape = (96, 96, 3) )
    input_2 = keras.layers.Input( shape = (96, 96, 3) )
    input_3 = keras.layers.Input( shape = (96, 96, 3) )

    model_1 = model(input_1)
    model_2 = model(input_2)
    model_3 = model(input_3)

    triplet_loss_layer = TripletLossLayer('relu')([model_1, model_2, model_3] )

    training_model = keras.Model(
        inputs = [input_1, input_2, input_3],
        outputs = triplet_loss_layer
    )

    opt = keras.optimizers.SGD()
    training_model.compile(optimizer=opt)
    
    return training_model

if __name__ == "__main__":
    args = sys.argv[1:]
    if args:
        if args[0].isdigit():
            training_model = fit_model()
            training_model.fit_generator(dataset(), epochs = int(args[0]), steps_per_epoch=10)




