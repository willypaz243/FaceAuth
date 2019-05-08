import tensorflow as tf

def triplet_loss(y_true, y_pred, alpha = 0.3):
    """
    Implementación de la "pérdida de triplete" 
    tal como se define en la fórmula (3) Argumentos:
    y_pred -- Lista de python que contiene 3 objetos:
            anchor -- las codificaciones de las imagenes de ancla, de shape (None, 128)
            positive -- las codificaciones de las imagenes, de shape (None, 128)
            negative -- las codificaciones de las imagenes, de shape (None, 128)

    Returns:
    loss -- un numero real, el valor de la perdida
    """
    anchor, positive, negative = y_pred[0], y_pred[1], y_pred[2]

    # Paso 1: Calcule la distancia (de codificación)
    # entre el ancla y el positivo, tendrá que sumar sobre eje = -1
    pos_dist = tf.reduce_sum(
        tf.square(
            tf.subtract(
                anchor,
                positive
            )
        ),
        axis=-1
    )
    
    # Paso 2: Calcule la distancia (de codificación) 
    # entre el ancla y el negativo, tendrá que sumar sobre el eje = -1    
    neg_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, negative)), axis=-1)
    # Paso 3: restar las dos distancias anteriores y agregar alfa.
    basic_loss = tf.add(tf.subtract(pos_dist, neg_dist), alpha)
    # Paso 4: Toma el máximo de basic_loss y 0.0. 
    # Suma sobre los ejemplos de entrenamiento.
    loss = tf.reduce_sum(tf.maximum(basic_loss, 0.0))

    return loss
