from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout, Flatten
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras import regularizers, initializers
from keras.optimizers import sgd

def uniform(scale):
        return initializers.uniform(minval=-scale, maxval=scale)

def normal(stdev):
        return initializers.normal(stddev=stdev)
def constant(const):
        return initializers.constant(value=const)
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
    Original implementation in pylearn2 can be found here: 
    https://github.com/karoldvl/paper-2015-esc-convnet/blob/master/Code/_Networks/Net-DoubleConv.ipynb
    """
    l2_param = 0.01
    model = Sequential()
    model.add(Conv2D(80, kernel_size=(input_dim[0] - 3, 6), strides=(1, 1),
                     activation=activation,
                     input_shape=input_dim,
                     kernel_initializer=uniform(0.001),
                     bias_initializer=constant(0.1),
                     kernel_regularizer=regularizers.l2(l2_param)))

    model.add(MaxPooling2D(pool_size=(4, 3), strides=(1, 3)))
    model.add(Dropout(0.75))

    model.add(Conv2D(80, kernel_size=(1, 3), strides=(1, 1),
                     activation=activation,
                     kernel_initializer=uniform(0.1), 
                     kernel_regularizer=regularizers.l2(l2_param)))

    model.add(MaxPooling2D(pool_size=(1, 3), strides=(1, 3)))

    model.add(Flatten())

    # Layer 1
    model.add(Dense(5000, kernel_initializer=normal(0.01), kernel_regularizer=regularizers.l2(l2_param)))
    model.add(Activation('relu'))
    model.add(Dropout(0.75))

    # layer 2
    model.add(Dense(5000, kernel_initializer=normal(0.01), kernel_regularizer=regularizers.l2(l2_param)))
    model.add(Activation('relu'))
    model.add(Dropout(0.75))

    # layer
    model.add(Dense(output_dim, kernel_initializer=normal(0.01), kernel_regularizer=regularizers.l2(l2_param)))
    model.add(Activation('softmax'))

    if input_dim[1] == 41:
        optimizer = sgd(lr=0.002, momentum=0.9, nesterov=True)
    else:
        if input_dim[1] == 101:
            optimizer = sgd(lr=0.01, momentum=0.9, nesterov=True)
        else:
            raise ValueError("Segment must either have 41 or 101 frames")

    model.compile(loss=loss,
                  optimizer=optimizer,
                  metrics=[metrics])

    return model
