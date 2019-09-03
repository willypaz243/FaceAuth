import os
import cv2
import numpy as np

PATH = os.path.abspath('./dataset')
PATH_RE = os.path.join(PATH, 'real_images')
PATH_POS = os.path.join(PATH, 'pos_images')
PATH_NEG = os.path.join(PATH, 'neg_images')


def dataset():
    while True:
        ancla = load_images(PATH_RE)
        positivo = load_images(PATH_POS)
        negativo = load_images(PATH_NEG)
        lista = [len(ancla), len(positivo), len(negativo)]
        yield [ancla[:min(lista)],
               positivo[:min(lista)],
               negativo[:min(lista)]], None

def load_images(path):
    images = [cv2.imread(path+'/'+arch.name) for arch in os.scandir(path) if arch.is_file()]
    images = np.array(images)
    images = np.resize(images, (images.shape[0], 96, 96, 3)) / 255
    return images
    
