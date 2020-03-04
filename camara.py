import cv2
import dlib
import numpy as np

from face_auth.FaceDetector import FaceDetector
from face_auth.ImgProcessor import ImgProcessor
from face_auth.FaceEncoder import FaceEncoder
from face_auth.face_model import face_model
from face_auth.kmean import K_mean
from face_auth.Identifier import Identifier

predictor = dlib.get_frontal_face_detector()
face_detector = FaceDetector(predictor)

landmark_detector = dlib.shape_predictor("face_auth/datas/shape_predictor_68_face_landmarks.dat")
img_processor = ImgProcessor(landmark_detector)

model = face_model()
model.load_weights("face_auth/models/model.h5")
face_encoder = FaceEncoder(model)

k_mean = K_mean(model_name="testing")

identifier = Identifier(face_encoder, k_mean, "test_identifier")

cap = cv2.VideoCapture()

def register(id_user):
    #activo = cap.open("videoplayback.mp4")
    #activo = cap.open("http://192.168.1.243:4040/video")
    activo = cap.open(0)
    face_images = []
    while activo:
        done, frame = cap.read()
        scale, frame = img_processor.rescale_img(frame, resolution=720)
        faces = face_detector(frame)
        
        if len(faces) == 1:
            face = faces[0]
            margin = np.round(face * 0.15).astype('int') * [[-1],[1]]
            face += margin
            (x1, y1),(x2, y2) = face
            if face.min() >= 0:
                face_img = frame[y1: y2, x1: x2]
                face_img = img_processor(face_img, margin)
                face_images.append(face_img)
                
            cv2.rectangle(frame, (x1,y1), (x2, y2), (0,255,0))
        else:
            print("Solo debe haber un rostro en la imagen para el registro.", end='\r')
        
        if len(face_images) > 30:
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(
                img=frame,
                text="Ya tenemos fotos suficientes para el registro precione 'q'",
                org=(10,30),
                fontFace=font,
                fontScale=0.6,
                color=(255,0,0),
                thickness=2
            )
            face_images.remove(face_images[0])
        
        if done:
            cv2.imshow('registor', frame)
            if cv2.waitKey(1) == ord('q'):
                cv2.destroyAllWindows()
                identifier.register(id_user, face_images)
                activo = False
    cap.release()

def identify():
    activo = cap.open(0)
    while activo:
        done, frame = cap.read()
        scale, frame = img_processor.rescale_img(frame, resolution=720)
        faces = face_detector(frame)
        
        for face in faces:
            margin = np.round(face * 0.15).astype('int') * [[-1],[1]]
            face += margin
            (x1, y1),(x2, y2) = face
            cv2.rectangle(frame, (x1,y1), (x2, y2), (0,255,0))
            if face.min() >= 0:
                face_img = frame[y1: y2, x1: x2]
                face_img = img_processor(face_img, margin)
                id_user = identifier.identify([face_img])
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(
                    img=frame,
                    text=str(id_user),
                    org=(x1-10,y1-30),
                    fontFace=font,
                    fontScale=0.6,
                    color=(255,0,0),
                    thickness=2
                )
        cv2.imshow('registor', frame)
        if cv2.waitKey(1) == ord('q'):
            cv2.destroyAllWindows()
            activo = False
cap.release()
            
    

if __name__ == "__main__":
    activo = True
    print("Bienvenido a este sistema de reconocimiento facial.")
    while activo:
        msg  = "Registrar   (r) \n" 
        msg += "identificar (i) \n" 
        msg += "salir       (s)\n\n" 
        msg += "entrada >>>  "
        entrada = input(msg)
        if entrada == "r":
            id_user = input("introduzca un id_user (ej:1234) :")
            while not id_user.isnumeric():
                id_user = input("debe de introducir un número (ej:1234) :")
            register(int(id_user))
        elif entrada == "i" :
            print("Presione la tecla 'q' para salir de la camara")
            identify()
        elif entrada == "s":
            activo = False
            
            
        
    
    
        
    

