import tensorflow as tf

@tf.function
def triplet_loss(y_true, y_pred, alpha = 0.3):
    """
    Implementación de la "pérdida de triplete" 
    tal como se define en la fórmula (3) Argumentos:
    y_pred -- Lista de python que contiene 3 tensores:
            anchor -- las codificaciones de las imagenes de ancla, de shape (None, 128)
            positive -- las codificaciones de las imagenes, de shape (None, 128)
            negative -- las codificaciones de las imagenes, de shape (None, 128)

    Returns:
    loss -- un numero real, el valor de la perdida
    """
    anchor, positive, negative = tuple(y_pred)

    # Paso 1: Calcule la distancia (de codificación)
    # entre el ancla y el positivo, tendrá que sumar sobre eje = 1
    pos_dist = ((anchor - positive)**2)**0.5
    pos_dist = tf.reduce_sum(pos_dist, axis=1)
    
    # Paso 2: Calcule la distancia (de codificación) 
    # entre el ancla y el negativo, tendrá que sumar sobre el eje = 1 
    neg_dist = ((anchor - negative)**2)**0.5
    neg_dist = tf.reduce_sum(neg_dist, axis=1)
    # Paso 3: restar las dos distancias anteriores y agregar alfa.
    basic_loss = (pos_dist - neg_dist) + alpha
    # Paso 4: Toma el máximo de basic_loss y 0.0. 
    # Suma sobre los ejemplos de entrenamiento.
    loss = tf.maximum( basic_loss, 0.0 )
    return loss


class TripletLossLayer(tf.keras.layers.Layer):
    def __init__(self, alpha, **kwargs):
        super(TripletLossLayer, self).__init__(**kwargs)
    def call(self, inputs):
        loss = triplet_loss(y_true = None, y_pred=inputs)
        self.add_loss(loss)
        return loss