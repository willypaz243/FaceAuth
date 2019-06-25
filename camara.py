import cv2 as cv2
import numpy as np
import keras
import os
from triplet_loss import triplet_loss
from k_mean import K_mean
from utils import prepare_k_means, get_most_similar, get_img_code, straighten_image,\
    get_eyes_centers, load_centroides, create_database, pd

print('Cargando modelo de reconicimiento facial')
MODEL = keras.models.load_model('my_model.h5')
MODEL.compile(optimizer = 'adam', loss = triplet_loss, metrics = ['accuracy'])

print('Preparando el modelo K-means')
MODEL_K = K_mean()
prepare_k_means(MODEL_K)

CAP = cv2.VideoCapture()
FACE_CASCADE = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
EYE_CASCADE = cv2.CascadeClassifier('haarcascade_eye.xml')

print('LISTO!')

def camara(ip=None):
    if ip == None:
        active = CAP.open(0)
    else:
        active = CAP.open(ip+'/video')
    while active:
        _, frame = CAP.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = FACE_CASCADE.detectMultiScale(
            gray,
            scaleFactor=1.2,
            minNeighbors=6,
            minSize=(96,96),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        for x,y,w,h in faces:
            x1 = x
            y1 = y
            x2 = x+w
            y2 = y+h
            parte_gray = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
            res, centros = get_eyes_centers(parte_gray, x, y)
            if res:
                if centros[0][0] < x+w/2 and centros[1][0] > x+w/2:
                    img = straighten_image(frame[y1:y2, x1:x2], eje=(x+w/2,y+h/2), puntos=centros)
                    img = cv2.resize(img,(104,104))
                    x_1 = 4
                    y_1 = 4
                    x_2 = 100
                    y_2 = 100

                    identidad = get_most_similar(get_img_code(img[y_1:y_2, x_1:x_2], MODEL)[0], MODEL_K)
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    cv2.putText(frame, identidad, (x1,y1-20), font, 1,(0,0,255),2,cv2.LINE_AA)
                cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0))
        cv2.imshow('CAMARA', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    CAP.release()
    cv2.destroyAllWindows()


def register_camera(nombre, ip = None):
    nombre  = nombre.upper()
    if ip == None:
        active = CAP.open(0)
    else:
        active = CAP.open(ip+'/video')
    while active:
        _, frame = CAP.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = FACE_CASCADE.detectMultiScale(
            gray,
            scaleFactor=1.2,
            minNeighbors=6,
            minSize=(96,96),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        muestras = []
        if len(faces) == 1 and len(muestras) <= 30:
            faces = faces[0]
            x1 = faces[0]
            y1 = faces[1]
            x2 = faces[0]+faces[2]
            y2 = faces[1]+faces[3]
            eje = (faces[0]+faces[2]/2,faces[1]+faces[3]/2)
            parte_gray = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
            res, centros = get_eyes_centers(parte_gray, x1, y1)
            if res and centros[0][0] < eje[0] and centros[1][0] > eje[1]:
                img = straighten_image(frame[y1:y2, x1:x2], eje=eje, puntos=centros)
                img = cv2.resize(img,(104,104))
                x_1 = 4
                y_1 = 4
                x_2 = 100
                y_2 = 100
                muestras.append(get_img_code(img[y_1:y_2, x_1:x_2], MODEL)[0])
            cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0))
        cv2.imshow('Camara', frame)
        dataset = np.array(muestras)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            centroides = load_centroides()
            centroides[nombre] = dataset.mean(axis=0)
            create_database(dataset) # Guarda los datos de las nuevas imagenes
            MODEL_K.set_centroides(np.array(list(centroides.values())))
            MODEL_K.set_nombre(list(centroides.keys()))
            dataframe = pd.DataFrame(centroides)
            if not os.path.exists("centroides"):
                os.mkdir("centroides")
            dataframe.to_csv("centroides/centros.csv")
            break
    CAP.release()
    cv2.destroyAllWindows()
    return faces
