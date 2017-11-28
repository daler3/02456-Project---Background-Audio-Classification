from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout, Flatten
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras import regularizers
from keras.optimizers import sgd


def piczak_CNN(input_dim, output_dim,
        activation='relu',
        metrics="accuracy", loss='categorical_crossentropy'):
    """
    This method returns a keras model describing Piczak implementation.

    From Piczak:
    The first convolutional ReLU layer consisted of 80 filters
    of rectangular shape (57x6 size, 1x1 stride) allowing
    for slight frequency invariance. Max-pooling was applied
    with a pool shape of 4x3 and stride of 1x3
    """

    model = Sequential()
    model.add(Conv2D(80, kernel_size=(input_dim[0] - 3, 6), strides=(1, 1),
                     activation=activation,
                     input_shape=input_dim))
                     #kernel_regularizer=regularizers.l2(0.001)))

    model.add(MaxPooling2D(pool_size=(4, 3), strides=(1, 3)))
    model.add(Dropout(0.5))

    model.add(Conv2D(80, kernel_size=(1, 3), strides=(1, 1),
                     activation=activation,
                     input_shape=input_dim))

    model.add(MaxPooling2D(pool_size=(1, 3), strides=(1, 3)))

    model.add(Flatten())

    # Layer 1
    model.add(Dense(5000))
    model.add(Dropout(0.5))
    model.add(Activation('relu'))

    # layer 2
    model.add(Dense(5000))
    model.add(Dropout(0.5))
    model.add(Activation('relu'))

    # layer
    model.add(Dense(output_dim))
    model.add(Activation('softmax'))

    if input_dim[1] == 41:
        optimizer = sgd(lr=0.002, momentum=0.9, nesterov=True, decay=0.001)
    else:
        if input_dim[1] == 101:
            optimizer = sgd(lr=0.01, momentum=0.9, nesterov=True, decay=0.001)
        else:
            raise ValueError("Segment must either have 41 or 101 frames")

    model.compile(loss=loss,
                  optimizer=optimizer,
                  metrics=[metrics])

    return model
