import numpy as np

class FaceDetector():
    
    def __init__(self, detectorModel):
        
        self.__detectorModel = detectorModel
    
    def detect_face(self, image):
        faces = self.__detectorModel(image)
        faces = self.__faces_to_numpy(faces)
        return faces
    
    def detect_first_face(self, image):
        faces = self.detect_face(image)
        first_face = []
        max_area = 0
        for face in faces:
            area = (face[1] - face[0])
            
            if area.prod() > max_area: 
                max_area = area.prod()
                first_face = face
        return np.array(first_face)
    
    def __faces_to_numpy(self, faces):
        faces = list(faces)
        for i, face in enumerate(faces):
            face_box = np.array([[ face.left(),  face.top()   ],
                                 [ face.right(), face.bottom()]])
            faces[i] = face_box
        return np.array(faces)
    
    def __call__(self, image):
        return self.detect_face(image)
            