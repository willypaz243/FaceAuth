import cv2
import numpy as np

from face_identity.Identificador import Identificador
from face_identity.kmean import K_mean


print('Cargando modelo de reconicimiento facial')
identificador = Identificador(name = "scesi_auth")
<<<<<<< HEAD
=======
k_mean = K_mean('pruebas_1_km')
>>>>>>> new_IDF

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
        #active = CAP.open('videoplayback.mp4')
        #active = CAP.open('crespo_respuesta_a_jl.mp4')
    else:
        active = CAP.open(ip+'/video')
    input_image_codes = []
    nombre = "Detectando..."
    while active:
        _, frame = CAP.read()
        new_shape = tuple(np.array(list(reversed(frame.shape[:-1]))) // 2)
        frame = cv2.resize(frame, new_shape)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_boxes, encode_images = identificador.box_and_encode(frame)
        
        media = []
        for code in input_image_codes:
            media.append(np.median(code, axis=0))
        media = np.array(media)
        
        if len(input_image_codes) == 0 and len(encode_images) > 0:
            input_image_codes = list(encode_images.reshape((len(encode_images), 1, 128)))
            #print(input_image_codes)
        elif len(encode_images) == len(input_image_codes):
            
            for face_box, code in zip(face_boxes, encode_images):
                index = np.linalg.norm(media - code, axis=1).argmin()
                input_image_codes[index] = np.concatenate([ input_image_codes[index], [code] ])
                nombre = 'Detectando...'
                if len(input_image_codes[index]) > 10:
                    input_code = np.median(input_image_codes[index], axis=0)
                    #input_image_codes[index] = (input_image_codes[index] + input_code) / 2
                    input_image_codes[index] = input_image_codes[index][1:]
                    #print(input_code)
                    nombre = str(k_mean.that_class(input_code))
                (x1, y1), (x2, y2) = face_box
                cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0))
                cv2.putText(frame, str(nombre), (x1,y1+10), font, 1,(0,0,255), 2, cv2.LINE_AA)
        
        elif len(encode_images) < len(input_image_codes):
            indices = list(range(len(input_image_codes)))
            
            for face_box, code in zip(face_boxes, encode_images):
                index = np.linalg.norm(media - code, axis=1).argmin()
                _ = indices.pop(index)
                
            for index in indices:
                _ = input_image_codes.pop(index)
                
        
        else:
            distances = []
            for code in encode_images:
                distance = np.linalg.norm(media - code, axis=1).min()
                distances.append(distance)
            index = np.argmax(distances)
            new_code = encode_images[index].reshape((1, 128))
            input_image_codes.append(new_code)

<<<<<<< HEAD
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
            
            if len(input_images) > 30:
                nombre = identificador.identify(input_images)
                input_images = []
            cv2.putText(frame, str(nombre), (x1,y1+20), font, 1,(0,0,255), 2, cv2.LINE_AA)
            cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0))
=======
>>>>>>> new_IDF
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
<<<<<<< HEAD
    images = []
    mjs = 'Preciona "q" para registrar su rostro'
    print(mjs)
=======
    input_codes = []
    mjs = 'Preciona "q" para registrar su rostro'
    print(mjs)
    
>>>>>>> new_IDF
    while active and id_user.isnumeric():
        _, frame = CAP.read()
        frame[:50] = np.array([0, 0, 255])
        frame[:50, : int(frame.shape[1] * (len(input_codes) / 100))] = np.array([255,0,0])
        
        face_boxes, encode_images = identificador.box_and_encode(frame)
        
        if len(encode_images) == 1 and len(input_codes) < 100:
            input_codes.append(encode_images[0])
            (x1, y1), (x2, y2) = face_boxes[0]
            cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0))
        
        elif len(input_codes) >= 100:
            median = np.median(input_codes, axis=0)
            input_codes = list( (np.array(input_codes) + median) / 2 )[:99]
        
        cv2.imshow('camara', frame)
        if (cv2.waitKey(1) & 0xFF == ord('q')):
            k_mean.add_class(id_user, np.array(input_codes))
            mjs = f'Su registro se completo exitosamente con {id_user}'
            break
    print(mjs)
    CAP.release()
    cv2.destroyAllWindows()
    return res

if __name__ == "__main__":
<<<<<<< HEAD
    camara()
    #register_camera('1232')
    #camara('http://192.168.1.159:4747')
    #register_camera('1232', 'http://192.168.1.159:4747')
=======
    camara('http://192.168.1.159:4040')
    #camara()
    #register_camera('1234', 'http://192.168.1.159:4040')
>>>>>>> new_IDF
