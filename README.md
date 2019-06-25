# Identificacion Facial
Esta implementación es una version simplificada de este codigo: https://github.com/Skuldur/facenet-face-recognition
Posee la misma estructura del modelo de red neuronal que empleo Skudur pero con funciones de uso mas simplificados
para un mejor entendimiento del codigo.

# Requisitos
Antes de poder ejecutar el codigo es necesario tener los siguientes requisitos:
  - Tener instalado python 3.7
  - tensorflow, un framework de machine Learning.                                        https://www.tensorflow.org/
  - keras, un framework de modelos de redes neuronales.                                  https://keras.io/
  - opencv-python, una libreria para el procesamiento de imagenes.                       https://opencv.org/
  - numpy, un libreria de calculos matematicos.                                          https://www.numpy.org/
  - matplotlib, una libreria para realizar graficas útil para el estudio de datos.       https://matplotlib.org/
  
Recomiendo utilizar anaconda:                                                            https://www.anaconda.com/ 
  - Anaconda es un distribución libre y abierta​ de los lenguajes Python y R, 
    utilizada en ciencia de datos, y aprendizaje automático y ya contiene varias librerias como las que mencione antes.

# Uso
Se a agregado la funcion register_camera() 
Abra una terminal e ingrese al interprete de python3

>>> from camara import register_camera, camara

>>> register_camera('su nombre')

si tiene un camara inalambrica ingrese como segundo parametro la ip de su camara, por ejemplo "http://127.0.0.1:6000".
Abrira una ventana mostrando la imagen de su camara, pulsa "ESC" para cerrar y registrar su foto.

>>> camara() 

ejecute la camara igualmente sí tiene una camara inalambrica ingrese el ip como parametro.

Esto podria tardar unos minutos debido a que debe cargar los datos del modelo.

Para cerrar la ventada de la camara presione 'ESC'

