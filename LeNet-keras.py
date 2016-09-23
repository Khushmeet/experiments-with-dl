from keras.models import Sequential
from keras.layers import Convolution2D, Dense, Flatten, MaxPooling2D, Activation
import numpy as np

classes = 10

x_train = np.reshape(28, 28)

# LeNet
# INPUT => CONV => RELU => POOL => CONV => RELU => POOL => FC => RELU => FC

model = Sequential()
model.add(Convolution2D(20, 5, 5, border_mode='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

model.add(Convolution2D(50, 5, 5, border_mode='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

model.add(Flatten())
model.add(Dense(500))
model.add(Activation('relu'))

model.add(Dense(classes))
model.add(Activation('softmax'))

