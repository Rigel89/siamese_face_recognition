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
    def __init__(self, imgShape, name='SiameseModel', lambda_l2=0.01):
        super().__init__()
        self._name=name
        regu =tf.keras.regularizers.L2(l2=lambda_l2)

        self.inp = Input(shape=imgShape)
        self.c1 = Conv2D(64, (10,10), strides=(1, 1), padding="valid",
                         activation='relu', kernel_regularizer=regu)
        self.m1 = MaxPooling2D((2,2))

        self.c2 = Conv2D(128, (7,7), strides=(1, 1), padding="valid",
                          activation='relu', kernel_regularizer=regu)
        self.m2 = MaxPooling2D((2,2))

        self.c3 = Conv2D(128, (4,4), strides=(1, 1), padding="valid",
                          activation='relu', kernel_regularizer=regu)
        self.m3 = MaxPooling2D((2,2))

        self.c4 = Conv2D(256, (4,4), strides=(1, 1), padding="valid",
                          activation='relu', kernel_regularizer=regu)
        self.f1 = Flatten()
        self.fc1 = Dense(4096, kernel_regularizer=regu)
        self.fc2 = Dense(4096, activation='sigmoid', kernel_regularizer=regu)

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
        x = self.fc1(x)
        return self.fc2(x)
    
    def summary(self):
        model = Model(inputs=[self.inp], outputs=[self.call(self.inp)], name=self.name)
        return model.summary()

class L1Dist(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__()
        self._name = 'L1Distance'
    def call(self, input_embedding, output_embedding):
        return tf.math.abs(input_embedding - output_embedding)

#nn = SiameseFE(imgShape=(105,105,3))

def siamese_FM(siamese_conv_model, imgShape, lambda_l2=0.01):
    regu =tf.keras.regularizers.L2(l2=lambda_l2)
    # Creating input shape
    input_image = Input(name='Input image', shape=imgShape)
    val_image= Input(name='Validation image', shape=imgShape)

    # Passing the input to the siamese convolution module
    input1 = siamese_conv_model(input_image)
    input2 = siamese_conv_model(val_image)

    # Calculating the L1 distance between the images
    siamese_L1dist = L1Dist()
    distance = siamese_L1dist(input1, input2)

    #Creating a classifier comparing the distance between pictures
    classifier = Dense(1,activation='sigmoid', kernel_regularizer=regu)(distance)

    return Model(inputs=[input_image, val_image], outputs=[classifier], name='SiameseNN')


#%% Reguralarizer schedule

class Decay(tf.keras.callbacks.Callback):

  def __init__(self, l2, decay_epoch, decay_rate, max_decay_epoch):
    super().__init__()
    self.l2 = l2
    self.decay_epoch = decay_epoch
    self.decay_rate = decay_rate
    self.max_decay_epoch = max_decay_epoch

  def on_epoch_end(self, epoch, logs=None):
    global_epoch_recomp = self.params.get('epoch')
    p = global_epoch_recomp / self.decay_epoch
    self.l2.assign(tf.multiply(self.l2, tf.pow(self.decay_rate, p)))
     
# l2 = tf.Variable(initial_value=0.01, trainable=False)

# def l2_regularizer(weights):
#     tf.print(l2)
#     loss = l2 * tf.reduce_sum(tf.square(weights))
#     return loss

# model = tf.keras.Sequential()
# model.add(tf.keras.layers.Dense(1, kernel_regularizer=l2_regularizer))
# model.compile(optimizer='adam', loss='mse')
# model.fit(tf.random.normal((50,1 )), tf.random.normal((50,1 )), batch_size=4, callbacks=[Decay(l2,
#     decay_steps=100000,
#     decay_rate=0.56,
#     staircase=False)], epochs=3)