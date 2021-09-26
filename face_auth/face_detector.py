import dlib
import numpy as np

class FaceDetector():
    
    def __init__(self, detectorModel=dlib.get_frontal_face_detector()):
        
        self.__detectorModel = detectorModel
    
    def detect_face_boxes(self, image):
        face_boxes = self.__detectorModel(image)
        face_boxes = self.__faces_to_numpy(face_boxes)
        return face_boxes
    
    def detect_first_face(self, image):
        face_boxes = self.detect_face_boxes(image)
        first_face = []
        max_area = 0
        for face in face_boxes:
            area = (face[1] - face[0])
            area_prod = area.prod()
            if area_prod > max_area: 
                max_area = area_prod
                first_face = face
        return np.array(first_face)
    
    def __faces_to_numpy(self, face_boxes):
        face_boxes = list(face_boxes)
        face_boxes = [
            [
                [ face.left(),  face.top()   ],
                [ face.right(), face.bottom()]
            ] for face in face_boxes
        ]
        return np.array(face_boxes)
    
    def __call__(self, image):
        return self.detect_face_boxes(image)
            