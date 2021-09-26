import cv2
import dlib
import numpy as np
from dlib import rectangle


class FaceImgProcessor:

    def __init__(self, landmarks_file):
        self.__landmarks_detector = dlib.shape_predictor(landmarks_file)

    def rescale_img(self, image, resolution):
        x, y = image.shape[:2]
        scale = resolution / y
        x = round(x * scale)
        y = resolution
        image = cv2.resize(image, (y, x))
        return scale, image

    def __get_landmarks(self, gray_img):
        x, y = gray_img.shape
        face = rectangle(0, 0, y, x)
        landmarks = self.__landmarks_detector(gray_img, face)
        landmarks = self.__landmarks_to_numpy(landmarks)
        return landmarks

    def __landmarks_to_numpy(self, landmarks):
        points = []
        for point in landmarks.parts():
            points.append([point.x, point.y])
        return np.array(points)

    def __calculate_angle(self, reference_points):
        punto1, punto2 = reference_points[:2]
        catetos = punto2-punto1
        radio = np.linalg.norm(catetos)
        angulo = np.degrees(np.arcsin(catetos[1] / radio))
        return angulo

    def __straighten_image(self, image, angle):
        x, y = image.shape[:2]
        centro = np.array(image.shape[:2]) // 2
        cateto = np.deg2rad(angle)
        cateto = int(centro.mean() * np.math.sin(cateto))
        giro = cv2.getRotationMatrix2D([int(c) for c in centro], angle, 1)
        image = cv2.warpAffine(image, giro, (x, y))
        return image

    def process_face_image(self, image, margin):
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        landmarks = self.__get_landmarks(gray_image)
        reference_points = [landmarks[36], landmarks[45]] # eyes location points
        angle = self.__calculate_angle(reference_points)
        image = self.__straighten_image(image, angle)
        (x1, y1), (x2, y2) = np.array([(0, 0), image.shape[:2]]) - margin
        image = image[y1: y2, x1: x2]

        return image

    def __call__(self, image, margin):
        return self.process_face_image(image, margin)
