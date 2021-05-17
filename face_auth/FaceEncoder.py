import cv2
import numpy as np


class FaceEncoder:
    
    def __init__(self, model):
        self.model = model
    
    def encode(self, face_img_inputs):
        face_img_inputs = self.__normalize_imputs(face_img_inputs)
        return self.__encode(face_img_inputs)
    
    def __normalize_imputs(self, face_img_inputs):
        inputs = []
        input_size = (96, 96)
        for img in face_img_inputs:
            img = cv2.resize(img, input_size)
            img = img / 255
            img = img.astype(np.float32)
            inputs.append(img)
        return np.array(inputs)
    
    def __encode(self, inputs):
        inputs = cv2.dnn.blobFromImages(inputs)
        self.model.setInput(inputs)
        predict = self.model.forward()
        return predict
    
    def __call__(self, face_img_inputs):
        return self.encode(face_img_inputs)