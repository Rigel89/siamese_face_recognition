# -*- coding: utf-8 -*-

#================================================================
#
#   File name   : architecture.py
#   Author      : Rigel89
#   Created date: 25/06/23
#   GitHub      : https://github.com/Rigel89/siamese_face_recognition
#   Description : siamese NN architecture script
#
#================================================================

#%% Importing libraries

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense


#%% Creating architecture

class SiameseFE(Model):
    def __init__(self, imgShape):
        super().__init__()
        self.inp = Input(shape=imgShape)
        self.c1 = Conv2D(64, (10,10), strides=(1, 1), padding="valid", activation='relu')
        self.m1 = MaxPooling2D((2,2))

        self.c2 = Conv2D(128, (7,7), strides=(1, 1), padding="valid", activation='relu')
        self.m2 = MaxPooling2D((2,2))

        self.c3 = Conv2D(128, (4,4), strides=(1, 1), padding="valid", activation='relu')
        self.m3 = MaxPooling2D((2,2))

        self.c4 = Conv2D(256, (4,4), strides=(1, 1), padding="valid", activation='relu')
        self.f1 = Flatten()
        self.fc = Dense(4096, activation='sigmoid')

    def call(self, inputs):
        #x = self.inp(inputs)
        x = self.c1(inputs)
        x = self.m1(x)

        x = self.c2(x)
        x = self.m2(x)

        x = self.c3(x)
        x = self.m3(x)

        x = self.c4(x)
        x = self.f1(x)
        return self.fc(x)
    
    def summary(self):
        model = Model(inputs=[self.inp], outputs=[self.call(self.inp)])
        return model.summary()

class L1Dist(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super.__init__()
    def call(self, input_embedding, output_embedding):
        return tf.math.abs(input_embedding - output_embedding)

nn = SiameseFE(imgShape=(105,105,3))
