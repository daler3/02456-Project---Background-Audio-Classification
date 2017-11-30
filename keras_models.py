from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout, Flatten
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras import regularizers
from keras import backend as K


# C(ˆ y t ,y t ) = −y t · log(ˆ y t ) − (1 − y t ) · log(1 − ˆ y t ) Loss function defined in the paper

def piczak_CNN(input_dim, output_dim,
			   activation='relu', optimizer="adam",
			   metrics="accuracy", loss='binary_crossentropy'):
	"""
	This method returns a keras model describing Piczak implementation.

	From Piczak:
	The first convolutional ReLU layer consisted of 80 filters
	of rectangular shape (57x6 size, 1x1 stride) allowing
	for slight frequency invariance. Max-pooling was applied
	with a pool shape of 4x3 and stride of 1x3
	"""

	model = Sequential()
	model.add(Conv2D(80, kernel_size=(57, 6,), strides=(1, 1),
					 activation=activation,
					 input_shape=input_dim,
					 kernel_regularizer=regularizers.l2(0.001)))

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
	# Removed the softmax output layer as a sigmoid transformation of the output is done by our custom loss function.
	#model.add(Activation('softmax'))
	model.add(Activation('sigmoid'))
	inp = model.input  # input placeholder
	outputs = [layer.output for layer in model.layers]  # all layer outputs
	functor = K.function([inp] + [K.learning_phase()], outputs)  # evaluation function

	print(outputs)
	print(functor)

	model.compile(loss=loss,
				  optimizer=optimizer,
				  metrics=[metrics])

	print("Model built")

	return model
