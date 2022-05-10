import tensorflow as tf

from tensorflow.keras import datasets, layers, models
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D

def model():
    model1 = Sequential()
    model1.add(Conv2D(filters=32, kernel_size= 3,strides=(1,1), activation= 'relu', input_shape=(28, 28, 1)))
    model1.add(MaxPool2D((2,2)))
    model1.add(Conv2D(filters=32, kernel_size= 3, strides= (1,1), activation= 'relu'))
    model1.add(MaxPool2D((2,2)))
    model1.add(Flatten())
    model1.add(Dense(units=10, activation= 'softmax'))
    model1.compile(loss='categorical_crossentropy',optimizer=tf.optimizers.Adam(), metrics=['accuracy'])

    return model1