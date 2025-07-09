#!/usr/bin/python
# -*- coding: utf-8 -*-
from __future__ import print_function

from keras import layers
from keras.models import Sequential, Model
from keras.layers import Conv2D, Conv2DTranspose
from keras.layers import Input, UpSampling2D
from keras.layers.normalization import BatchNormalization
from keras.layers import Conv2D, Conv2DTranspose
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dropout
from keras.layers.core import Dense
from keras import backend as K


# ----------------------------------------------------------------------------
def create_features_model(model):
    return Model(inputs = model.input,
                                outputs = model.get_layer('features').output)


# ----------------------------------------------------------------------------
def __get_batch_normalization_axis():
    if K.image_data_format() == 'channels_last':
        return -1
    return 1


#------------------------------------------------------------------------------
def get_model(input_shape, classes, categorical=False):
    nbaxis = __get_batch_normalization_axis()
    array_kernels = [(11, 11), (9,9), (7,7), (5, 5), (3,3)]
    array_filters = [32,   32,   64,   64,    128]
    array_dropout = [0.1, 0.2, 0.3, 0.4, 0.5, 0.5]

    model = Sequential()

    model.add(Conv2D(array_filters[0], array_kernels[0], padding='same', input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(BatchNormalization(axis=nbaxis))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(array_dropout[0]))

    model.add(Conv2D(array_filters[1], array_kernels[1], padding='same'))
    model.add(Activation('relu'))
    model.add(BatchNormalization(axis=nbaxis))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(array_dropout[1]))

    model.add(Conv2D(array_filters[2], array_kernels[2], padding='same'))
    model.add(Activation('relu'))
    model.add(BatchNormalization(axis=nbaxis))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(array_dropout[2]))

    model.add(Conv2D(array_filters[3], array_kernels[3], padding='same'))
    model.add(Activation('relu'))
    model.add(BatchNormalization(axis=nbaxis))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(array_dropout[3]))

    model.add(Conv2D(array_filters[4], array_kernels[4], padding='same'))
    model.add(Activation('relu'))
    model.add(BatchNormalization(axis=nbaxis))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(array_dropout[4]))

    model.add(Flatten())
    model.add(Dense(128, name='features'))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(array_dropout[5]))

    model.add(Dense(classes))

    if categorical:
        model.add(Activation('softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=["accuracy"])
    else:
        model.add(Activation('sigmoid'))        # for multi-label classification
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=["accuracy"])

    return model


# ----------------------------------------------------------------------------
def get_autoencoder(input_size):
    nbaxis = __get_batch_normalization_axis()
    kernel = (3, 3)
    strides = (2,2)
    dropout1 = 0.25

    input_img = Input(shape=input_size)

    x = Conv2D(128, kernel, padding='same', strides=strides)(input_img)
    x = BatchNormalization(axis=nbaxis)(x)
    x = Activation('relu')(x)
    e1 = Dropout(dropout1)(x)

    x = Conv2D(64, kernel, padding='same', strides=strides)(e1)
    x = BatchNormalization(axis=nbaxis)(x)
    x = Activation('relu')(x)
    e2 = Dropout(dropout1)(x)

    x = Conv2D(64, kernel, padding='same', strides=strides)(e2)
    x = BatchNormalization(axis=nbaxis)(x)
    x = Activation('relu')(x)
    e3 = Dropout(dropout1)(x)

    x = Conv2D(64, kernel, padding='same', strides=strides)(e3)
    x = BatchNormalization(axis=nbaxis)(x)
    x = Activation('relu')(x)
    e4 = Dropout(dropout1)(x)

    encoded = Conv2D(1, kernel, padding='same', strides=1, name='features')(e4)

    x = Conv2DTranspose(64, kernel, padding='same', strides=strides)(encoded)
    x = BatchNormalization(axis=nbaxis)(x)
    x = Activation('relu')(x)
    x = Dropout(dropout1)(x)
    x = layers.concatenate([x, e3])

    x = Conv2DTranspose(64, kernel, padding='same', strides=strides)(x)
    x = BatchNormalization(axis=nbaxis)(x)
    x = Activation('relu')(x)
    x = Dropout(dropout1)(x)
    x = layers.concatenate([x, e2])

    x = Conv2DTranspose(64, kernel, padding='same', strides=strides)(x)
    x = BatchNormalization(axis=nbaxis)(x)
    x = Activation('relu')(x)
    x = Dropout(dropout1)(x)
    x = layers.concatenate([x, e1])

    x = Conv2DTranspose(128, kernel, padding='same', strides=strides)(x)
    x = BatchNormalization(axis=nbaxis)(x)
    x = Activation('relu')(x)
    x = Dropout(dropout1)(x)
    x = layers.concatenate([x, input_img])

    decoded = Conv2D(3, kernel, strides=1, activation='sigmoid', padding='same')(x)

    autoencoder = Model(input_img, decoded)

    autoencoder.compile(optimizer='adam', loss='binary_crossentropy', metrics=['mse'])

    return autoencoder


