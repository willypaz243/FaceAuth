import cv2
from id_model import faceRecoModel
from triplet_loss import triplet_loss
from utils import load_weights_from_FaceNet, load_database, get_img_code, get_most_similar

# Creamos un nuevo modelo
model = faceRecoModel(input_shape=(3, 96, 96))
# Creamos una base de datos con la imagenes del directorio 'images/'
database = load_database(model)
# Fuente del texto para la imagen
font = cv2.FONT_HERSHEY_SIMPLEX

# Copila en modelo
model.compile(optimizer = 'adam', loss = triplet_loss, metrics = ['accuracy'])
print('Cargando pesos predefinidos....')
# Carga los pesos de las capas
load_weights_from_FaceNet(model)

# Abrimos una captura de opencv
CAP = cv2.VideoCapture()
# Cargamos Clasificador en cascada que reconocera rostros humanos.
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
print(face_cascade.empty(), "Pesos cargados")

def camara(ip = None):
    """
    Implementacion de una camara web para identificar 
    personas con sus rostros.

    Puede usar una camara wifi o por IP descomentando las siguiente linea
    Reemplace el IP de muestra con el IP de su camara.

    """
    if ip == None:
        active = CAP.open(0) and not face_cascade.empty()
    else:
        active = CAP.open(ip+'/video') and not face_cascade.empty()
    while active:
        ret, frame = CAP.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        #gray = cv2.equalizeHist(gray)

        # Detecta los limites de un rostro y los almacena en una lista
        faces = face_cascade.detectMultiScale(gray, 1.2, 3)
        for (x, y, w, h) in faces:
            # Fijando dimenciones
            x1 = x
            y1 = y
            x2 = x+w
            y2 = y+h
            tam_y = y2-y1
            tam_y = (tam_y//100)*20
            y2 += tam_y
            # Extrae el fracmento de la imagen original que posee el rostro
            parte = frame[y1:y2, x1:x2]
            # Identifica a la persona y estrae su nombre de la foto
            identidad = get_most_similar(get_img_code(parte, model), database)
            # Agraga un texto con el nombre del identificado al la imagen
            frame = cv2.putText(frame, identidad, (x1,y1+20), font, 1,(0,0,255),2,cv2.LINE_AA)
            # Enmarca el rostro del usuario identificado
            frame = cv2.rectangle(frame,(x1, y1),(x2, y2),(255,0,0),2)
        cv2.imshow('Camara', frame)
        key = cv2.waitKey(1)
        if key == 27: # Preciona Esc para Salir
            break
    CAP.release()
    cv2.destroyAllWindows()


