from tensorflow import keras

def create_model():
    input = keras.layers.Input(shape = ( 96, 96, 3 ))

    hidden_layer = keras.layers.Conv2D(
        filters=32,
        kernel_size=(3, 3),
        padding="same",
        activation="relu"
    )(input)
    hidden_layer = keras.layers.MaxPool2D(pool_size=(2, 2), strides=2)(hidden_layer)
    hidden_layer = keras.layers.Dropout(0.5)(hidden_layer)

    hidden_layer = keras.layers.Conv2D(
        filters=32,
        kernel_size=(3, 3),
        padding="same",
        activation="relu"
    )(input)
    hidden_layer = keras.layers.MaxPool2D(pool_size=(2, 2), strides=2)(hidden_layer)
    hidden_layer = keras.layers.Dropout(0.5)(hidden_layer)

    hidden_layer = keras.layers.Flatten()(hidden_layer)
    output = keras.layers.Dense(128, activation="softmax")(hidden_layer)

    return keras.Model(inputs=input, outputs=output)
