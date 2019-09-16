import cv2
import numpy as np

from face_identity.Identificador import Identificador


print('Cargando modelo de reconicimiento facial')
identificador = Identificador(name = "prueba")

FACE_CASCADE = cv2.CascadeClassifier('face_identity/haarcascade_frontalface_default.xml')
EYE_CASCADE = cv2.CascadeClassifier('face_identity/haarcascade_eye.xml')

print('LISTO!')

CAP = cv2.VideoCapture()

def camara(ip=None):
    """
    Mientras se captura las imagenes de una camara, esta función detecta los
    rostros de cada frame y los identifica.
    
    Args:
        - ip: En el caso de que se use una camara mediante wifi,
              recibe una entrada de esta forma "http://127.0.0.1"
              o el ip de su dispositivo.
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    if ip == None:
        active = CAP.open(0)
    else:
        active = CAP.open(ip+'/video')
    input_images = []
    nombre = "Detectando..."
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
            input_images.append(frame)
            if len(input_images) > 5:
                nombre = identificador.identify(input_images)
                input_images = []
            cv2.putText(frame, str(nombre), (x1,y1+10), font, 1,(0,0,255), 2, cv2.LINE_AA)
            cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0))
        cv2.imshow('CAMARA', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    CAP.release()
    cv2.destroyAllWindows()


def register_camera(id_user, ip = None):
    """
    Sirve para registrar a un usuario al cual identificarlo mediante su rostro.
    
    Args:
        - id_user: debe ser un número por el cual se identificara
                   a una persona.
                   
        - ip: En el caso de que se use una camara mediante wifi,
              recibe una entrada de esta forma "http://127.0.0.1"
              o el ip de su dispositivo.
    """
    mjs = "id_user debe ser un número"
    res = False
    if ip == None:
        active = CAP.open(0)
    else:
        active = CAP.open(ip+'/video')
    images = []
    while active and id_user.isnumeric():
        mjs = 'Preciona "q" para registrar su rostro'.
        print(mjs)
        _, frame = CAP.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        faces = FACE_CASCADE.detectMultiScale(
            gray,
            scaleFactor=1.2,
            minNeighbors=6,
            minSize=(96,96),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        if len(faces) == 1:
            faces = faces[0]
            x1 = faces[0]
            y1 = faces[1]
            x2 = faces[0]+faces[2]
            y2 = faces[1]+faces[3]
            cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0))
            images.append(frame)
            if len(images) > 30:
                images.remove(images[0])
        cv2.imshow('Camara', frame)
        if (cv2.waitKey(1) & 0xFF == ord('q')):
            images = np.array(images)
            print(images.shape)
            res = identificador.registrar_usuario(id_user, images)
            mjs = 'Su registro se completo exitosamente'
            break
    print(mjs)
    CAP.release()
    cv2.destroyAllWindows()
    return res
