
# Identificacion-Facial
Buenas, Este es un proyecto personal en el que basandome en el paper [FaceNet: A Unified Embedding for Face Recognition and Clustering](https://arxiv.org/pdf/1503.03832.pdf)
Implemente un programa para acceso mediante la identificación facial.

## ¿Cómo funciona?
Para identificar a una persona o registrar a un grupo de personas a las que se quiere registrar, el programa realiza los siguientes pasos.

### 1. Detectar un rostro.
  Antes de identificar un rostro en una imagen debemos saber donde está, para hacer esto utilizando de la libreria [opencv](https://opencv.org/)
  usamos la funciones de [haar-cascade](https://docs.opencv.org/master/db/d28/tutorial_cascade_classifier.html) para ubicar la posicion del rostro con respecto a la imagen completa.
  Con esto podemos hacer un recorte de la imagen en donde solo este el rostro.

### 2. Procesar la imagen.
  Para evitar que los datos que nos puede proporcionar la imagen de un rostro varíen como puede ser, la inclinación de la cabeza,
  la iluminación, la calidad de la imagen, etc. Debemos procesar la imagen de tal forma en que esos datos no tengan esa variacion.
  Algunos de esos proceso pueden ser:
  - Enderezar la imagen con respecto a los ojos para que estos esten a la misma altura con respecto a la imagen.(Es recomendable hacerlos antes del recorte).
  - Realizar una [ecualización](https://docs.opencv.org/4.1.1/d6/dc7/group__imgproc__hist.html#ga7e54091f0c937d49bf84152a16f76d6e)
  de la imagen para reducir el ruido que puede causar la iluminación.

### 3. Codificar la imagen.
  Despues de realizar todo este procesamiento a la imagen de un rostro ya podemos ingresarla a una red neuronal, 
  en el [paper](#Identificacion-Facial) mencionado al inicio nos proporciona la arquitectura de una red neuronal
  adecuada para este trabajo, este modelo nos da una salida de una matriz de (128,) que sera el codigo de la imagen.
  Esta codificación de la imagen debemos asociarlo a una identidad, pero para un mismo rostro esta codificación variará minimamente,
  por lo que la identidad deberia asociarse a un grupo de matrices de (128,) con esta minima variación.
  
### 4. Agrupar los datos.
 Finalmente para realizar la identificación primero debemos registrar a quienes vamos identificar, para esta tarea me propuso a utilizar
 un modelo de aprendizaje no-supervizado, el modelo [k-means](https://es.wikipedia.org/wiki/K-medias) en el cual dado un número determinado
 de clases 'k' agrupar un conjunto de datos en 'k' grupos.
 
 - **Registro**.- De esta forma podemos registrar a una persona con un conjunto de imagenes de su rostro no menos de 10 imagenes,
 pasarlos por la red neuronal para obtener un conjunto de codificaciones, sacar la media de todos esos datos, que resultará en una matriz (128,)
 el cual servirá como punto de referencia para su identificación y que sera el primer grupo del modelo **k-means**, a medida que registremos a
 mas personas el valor k se incrementara a la candidad de personas registradas.
 - **Identificación**.- Usando una imagen o conjunto de imagenes del rostro que se va a determinar su identidad, pasando por la red neuronal
 la codificación que resulte se determinará la identidad usando k-means para comparar a que grupo se parece mas, mediante eso, el programa nos dira a quien estan identificando.
 
 ## Entrenamiento.
  Aquí hay dos modelos que entrenar.
  - **facenet**: Es el modelo que codifica la imagen en 128 datos por imagen, en su entrenamiento se utiliza la funcion de perdida
  [triplet-loss](https://en.wikipedia.org/wiki/Triplet_loss) el cual consiste en evaluar 3 resultados, en este caso los datos codificados de 3 imagenes,
  una de la imagen verdadera, la de una imagen parecida, y otra diferente, el algoritmo evalua la distancia de la imagen verdadera(objetivo o target),
  a las imagen parecida (positivo) y la que es diferente(negativo), y obtimisara la red neuronal para que la distancia entre el objetivo y el positivo sea
  mas corta y la distancia entre el objetivo y el negativo sea mas amplia. Hay que mencionar que para tener un modelo robusto,
  es necesario entrenar la red neuronal con una cantidad de datos que superan al millón, afortunadamente ya existen modelos preentrenados,
  con la arquitectura que estamos manejando, en mi caso los saqué de tensorflow, en formato de archivos CSV.
  
  - **[k-means](https://es.wikipedia.org/wiki/K-medias)**: El entrenamiento de este modelo es sencillo y rapido, en nuestro caso el entrenamiento inicia en el momento en el que
  se registra a una primera persona, la idea es agrupar un conjunto de datos en 'k' grupos, para esto se usas 'k' puntos referenciales
  que se denominan 'centroides', que se irán moviendo hacia la media de los datos mas cercanos a ellos, este proceso se repite hasta que
  los centroides sean la media de los datos mas cercanos a si mismo.
  
  ## Pre-requisitos.
  
    Para que el programa funcione correctamente debe instalar los paquetes especificados en el requirements.txt
    - En un entorno virtual con python=3.x
    ```
    pip install -r requirements.txt
    ```
    
    - Sin entorno virtual:
    ```
    pip3 install -r requirements.txt
    ```
  
  ## Lanzar el programa.
  Para provar el programa ejecute el el script camara.py

  En el entorno virtual con python3.x
  ```bash
  python camara.py
  ```
  Sin el entorno virtual.
  ```bash
  python3 camara.py
  ```
  se lanzará una aplicacion de consola con las siguinetes opciones.
  ```bash
  Presione la tecla 'q' para salir de la camara
  Registrar   (r)
  identificar (i)
  salir       (s)

  entrada >>>
  ```

  ## Autor
    * **Willy Samuel Paz Colque**
