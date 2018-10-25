import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import Adam
import numpy as np
import pandas as pd
from random import shuffle

WIDTH = 120
HEIGHT = 160

train_data = np.load('data_train.npy')

df = pd.DataFrame(train_data)
df = df.iloc[np.random.permutation(len(df))]
train_data = df.values.tolist()




X = np.array([i[0] for i in train_data]).reshape(-1,WIDTH,HEIGHT,1)
y = [i[1] for i in train_data]

X = tf.keras.utils.normalize(X, axis=1)


def geometrynet():
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(8, 8), strides=(4, 4),
                     activation='relu',
                     input_shape=[ WIDTH, HEIGHT, 1]))

    # model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(Conv2D(32, (4, 4)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))

    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))

    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))

    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    #
    # model.compile(loss='binary_crossentropy',
    #               optimizer='adam',
    #               learning_rate=0.01,
    #               metrics=['accuracy'],
    #               )

    # model.compile(optimizer='rmsprop',
    #           loss='binary_crossentropy',
    #           metrics=['accuracy'])

    model.compile(loss='binary_crossentropy',
                          optimizer='adam',
                          metrics=['accuracy'],
                          )

    model.fit(X, y,
    batch_size=16,
    epochs=10,
    validation_split=0.2,
    )

    model.save('model_test.h5')

    return model

geometrynet()
