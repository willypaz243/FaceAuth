import cv2
from id_model import faceRecoModel
from triplet_loss import triplet_loss
from utils import load_weights_from_FaceNet, load_database, get_img_code, get_most_similar

model = faceRecoModel(input_shape=(3, 96, 96))
database = load_database(model)
font = cv2.FONT_HERSHEY_SIMPLEX

model.compile(optimizer = 'adam', loss = triplet_loss, metrics = ['accuracy'])
print('Cargando pesos predefinidos....')
load_weights_from_FaceNet(model)

CAP = cv2.VideoCapture()
face_cascade = cv2.CascadeClassifier('/home/sargeras243/opencv/data/haarcascades/haarcascade_frontalface_default.xml')
print(face_cascade.empty(), "Pesos cargados")
def camara():
    active = CAP.open(0) and not face_cascade.empty()
    #active = CAP.open('http://192.168.0.10:4747/video') and not face_cascade.empty()
    while active:
        ret, frame = CAP.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        faces = face_cascade.detectMultiScale(gray, 1.2, 3)
        for (x, y, w, h) in faces:
            x1 = x
            y1 = y
            x2 = x+w
            y2 = y+h#+(((y+h)-y)/100)*20
            tam_y = y2-y1
            tam_y = (tam_y//100)*20
            y2 += tam_y
            parte = frame[y1:y2, x1:x2]
            identidad = get_most_similar(get_img_code(parte, model), database)
            frame = cv2.putText(frame, identidad, (x1,y1+20), font, 1,(0,0,255),2,cv2.LINE_AA)
            frame = cv2.rectangle(frame,(x1, y1),(x2, y2),(255,0,0),2)
        cv2.imshow('Camara', frame)
        key = cv2.waitKey(1)
        if key == 27:
            #cv2.imwrite('images/Willy.jpg', parte)
            break
    CAP.release()
    cv2.destroyAllWindows()


