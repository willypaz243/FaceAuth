import os
import cv2
import dlib
import numpy as np

path = os.path.abspath('shape_predictor_68_face_landmarks.dat')
DETECTOR = dlib.get_frontal_face_detector()
PREDICTOR = dlib.shape_predictor(path)

def get_eyes_points(img, face):
    gray = np.uint8(np.round(np.mean(img, axis=2)))
    landmarks = PREDICTOR(gray, face)
    landmarks = np.array(landmarks.parts())
    right_eye = landmarks[36:42].mean()
    left_eye = landmarks[42:47].mean()
            
    eye_points = [[ right_eye.x ,right_eye.y ],
                  [ left_eye.x  ,left_eye.y  ]]
    
    return np.array(eye_points).astype('int')

def get_frontal_face(img):
    gray = np.uint8(np.round(np.mean(img, axis=2)))
    faces = DETECTOR(gray)
    area = 0
    rostro = None
    for face in faces:
        if face.area() > area:
            rostro = face
    return rostro

def enderezar_imagen(img, angulo):
    x, y = img.shape[: -1]
    centro = np.array(img.shape[: -1]) // 2
    cateto = np.deg2rad(angulo)
    cateto = int(centro.mean() * np.math.sin(cateto))
    #print(centro, angulo, '<------------------')
    giro = cv2.getRotationMatrix2D(tuple(centro), angulo, 1)
    img = cv2.warpAffine(img, giro, (x, y))
    img = img[abs(cateto): y-abs(cateto), abs(cateto): x-abs(cateto)]
    return img

def calcular_angulo(reference_points):
    punto1, punto2 = reference_points
    catetos = punto2-punto1
    hipotenusa = np.linalg.norm(punto1-punto2)
    angulo = np.degrees(np.math.asin(catetos[1] / hipotenusa))
    return angulo
    
    
def obtener_recorte(img, face):
    aumento = abs(round((face.left() - face.top()) * 0.1))
    x1, y1 = np.array([face.left(), face.top()]) + aumento
    x2, y2 = np.array([face.right(), face.bottom()]) + aumento
    img = img[y1: y2, x1: x2]
    return img


def procesar_imagenes(images):
    imgs = []
    for img in images:
        face = get_frontal_face(img)
        if face is not None:
            eye_points = get_eyes_points(img, face)
            img_face = obtener_recorte(img, face)
            angulo_rotacion = calcular_angulo(eye_points)
            img_face = enderezar_imagen(img_face, angulo_rotacion)
            img_face = cv2.resize(img_face, (96,96))
            img_face = img_face / 255
            imgs.append(img_face)
    return np.array(imgs)