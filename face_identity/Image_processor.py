import cv2
import dlib
import numpy as np
import tensorflow as tf

predictor = dlib.get_frontal_face_detector()
detector = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

def rescale_image(image):
    """
    Escala la imagen a un tamaño tal que su anchura sea de 512px.
    Args:
        -image: una imagen de tipo matriz NumPy.
    Returns:
        Retorna la imagen escalada.
    """
    x, y = image.shape[:2]
    scale = 512 / y
    x = round(x * scale)
    y = 512
    image = cv2.resize(image, (y,x))
    return scale, image

def get_landmarks(gray):
    """
    Obtiene puntos LandMarks de una imagen que contenga un solo rostro.
    
    El rostro debe ocupar la mayor parte del área de la imagen.
    
    Los puntos LandMarks son puntos caracteristicos del rostro que nos sirve para hacer seguimiento de distintas partes del rostro como nariz, ojos, boca, etc.
    Args:
        -gray: una imagen a escala de grises de tipo matriz NumPy.
    Returns:
        -retorna una matriz NumPy de dimensiones (2, 68), es decir 68 puntos (x,y)
    """
    x,y = gray.shape
    face = dlib.rectangle(0, 0, y, x)
    landmarks = detector(gray, face)
    landmarks = landmarks_to_numpy(landmarks)
    return landmarks

def landmarks_to_numpy(landmarks):
    """
    Convierte los puntos LandMarks de tipo Points tipo de dato que retorna la libreria Dlib en una matriz NumPy.
    Arg:
        -landmarks: 68 puntos de tipo Points obtenidos de la libreria Dlib.
    Returns:
        -Retorna los mismos puntos en formato matriz NumPy.
    """
    points = []
    for point in landmarks.parts():
        points.append([point.x, point.y])
    return np.array(points)

def calcular_angulo(reference_points):
    punto1, punto2 = reference_points
    catetos = punto2-punto1
    radio = np.linalg.norm(catetos)
    angulo = np.degrees(np.arcsin(catetos[1] / radio))
    return angulo

def enderezar_imagen(image, angulo):
    x, y = image.shape[:2]
    centro = np.array(image.shape[:2]) // 2
    cateto = np.deg2rad(angulo)
    cateto = int(centro.mean() * np.math.sin(cateto))
    giro = cv2.getRotationMatrix2D(tuple(centro), angulo, 1)
    image = cv2.warpAffine(image, giro, (x, y))
    return image

def rotar_puntos(puntos, angulo, centro):
    """
    Mueve una serie de puntos (x,y) sobre un centro.
    Esta funcion se utiliza para rotar los puntos LandMarks que se extrajo de una imagen.
    
    Arg:
        - puntos: Una matriz NumPy de dimenciones (2,n) donde n la cantidad de puntos.
        - angulo: En angulo que rotaran los puntos sobre el eje centro.
        - centro: Es el punto de referencia sobre el que rotaran los puntos.
    Returns:
        - Retorna los puntos ya rotados en una matriz NumPy.
    """
    vectores = puntos - centro
    radios = np.linalg.norm(vectores, axis=1)
    x = vectores[:,0]
    y = vectores[:,1]
    
    
    angulos = 360*((y < 0) * (x > 0)) + 180 * ((x < 0) * (y < 0)) + 180 * ((x < 0) * (y > 0)) + np.degrees(np.arctan(y/( x + 1e-100)))
    angulos = np.deg2rad(angulos)
    puntos[:,0] = radios * np.cos(angulos)
    puntos[:,1] = radios * np.sin(angulos)
    puntos = puntos + centro
    radios = np.linalg.norm(centro-puntos, axis=1)
    #print(radios.max())
    return np.round(puntos).astype('int')

