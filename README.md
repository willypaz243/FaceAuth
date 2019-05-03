# Identificacion-Facial
Esta implementacion es una version simplificada de este codigo: https://github.com/Skuldur/facenet-face-recognition
Posee la misma estructura del modelo de red neuronal que empleo Skulur pero con funciones de uso mas simplificados
para un mejor entendimiento del codigo.

#Requisitos
Antes de poder ejecutar el codigo es necesario tener los siguientes requisitos:
  - Tener instalado python 3.7
  - tensorflow, un framework de machine Learning.                                        https://www.tensorflow.org/
  - keras, un framework de modelos de redes neuronales.                                  https://keras.io/
  - opencv-python, una libreria para el procesamiento de imagenes.                       https://opencv.org/
  - numpy, un libreria de calculos matematicos.                                          https://www.numpy.org/
  - matplotlib, una libreria util para realizar graficas util para el estudio de datos.  https://matplotlib.org/
  
Recomiendo utilisar anaconda:                                                            https://www.anaconda.com/ 
  - Anaconda es un distribución libre y abierta​ de los lenguajes Python y R, 
    utilizada en ciencia de datos, y aprendizaje automático y ya contiene varios de los requisitos que mencione antes.

#Uso
Primero agregue una foto suya donde solo se vea su rostro en la carpeta images/
luego puede abrir una terminal python e importar la funcion camara()de esta forma:

>>> from camara import camara
>>> camara() # con esto ya funcionaria la camara utilizara la primera camara conectada a su pc o la webcam 

Esto podria tardar unos minutos debido a que debe cargar los datos del modelo.
o tambien puede crear un archivo python3 y escribir el mismo codigo de la terminal.

