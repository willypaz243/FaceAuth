import cv2
import numpy as np

class Image_processor:

    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier('face_identity/haarcascade_frontalface_default.xml')
        self.eye_cascade = cv2.CascadeClassifier('face_identity/haarcascade_eye.xml')

    def cut_face(self, image):
        """
        Corta el recuadro donde se encuentre un rostro.
        
        Args:
            - image: Una imagen en forma de matriz de tipo numpy.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
        gray = cv2.equalizeHist(gray)
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.2,
            minNeighbors=6,
            minSize=(96, 96),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        faces = np.array(faces)
        cut_image = np.array([])
        if faces.any():
            x, y, w, h = faces[0]
            cut_image = image[y:y + h, x:x + w]
            cut_image = self.enderesar_imagen(cut_image)
        return cut_image
    
    def eyes_centers(self, image):
        """
        Encuentra los puntos centrales de los ojos en una imagen, 
        esto nos sirve para poder enderezar la imagen y evitar la
        variacion en la identificación.
        
        Args:
            - image: Una imagen en forma de matriz de tipo numpy.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        eyes = self.eye_cascade.detectMultiScale(
            gray,
            scaleFactor=1.2,
            minNeighbors=6,
            minSize=(16,16),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        centros = np.int32([])
        if len(eyes) >= 2:
            for x,y,w,h in eyes:
                centros = np.append(centros, [x + w // 2, y + h // 2])
            centros = np.reshape(centros, (centros.shape[0]//2,2))
        return centros

    def enderesar_imagen(self, image):
        """
        Usando los puntos donde se encuentran los ojos, enderesa la imagen para
        evitar la variacion de la imagen al momento de identificar al alguien
        mediante su foto.
        
        Args:
            - image: Una imagen en forma de matriz de tipo numpy.
        """
        centros = self.eyes_centers(image)
        imagen_enderezada = np.array([])
        if centros.any():
            x , y , _ = image.shape
            b1 = centros[:,0] < x / 2
            b2 = centros[:,0] > x / 2
            centros = np.int32([
                np.mean(np.delete(centros, np.where(b1==False), axis=0), axis=0),
                np.mean(np.delete(centros, np.where(b2==False), axis=0), axis=0)
            ])
            if centros[0][0] < x / 2 and centros[1][0] > x / 2:
                cateto = (centros[1][1] - centros[0][1])
                dist = ((centros[1] - centros[0]) ** 2).sum() ** 0.5
                angulo = np.degrees(np.math.asin(cateto / dist))
                m = cv2.getRotationMatrix2D((x // 2, y // 2), angulo, 1)
                image = cv2.warpAffine(image, m, (x, y))
                imagen_enderezada = image[abs(cateto): x - abs(cateto), abs(cateto): y - abs(cateto)]
        return imagen_enderezada

    def process_image(self, images):
        """
        Procesa un lote de imagenes facilitar la detección e identificación de rostros.
        
        Args:
            - images: Un lote imagenes en forma de matriz tipo numpy.
        """
        processed_images = []
        for image in images:
            image = self.cut_face(image)
            if image.any():
                for i in range(3):
                    image[:, :, i] = cv2.equalizeHist(image[:, :, i])
                image = 255 - image
                image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
                image = cv2.resize(image, (96, 96))
                image = np.float32(image / 255)
                processed_images.append(image)
        return np.float32(processed_images)