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
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Concatenate


#%% Creating architecture

class InceptionV2(Model):
    def __init__(self,filters=256, name='Inception_V2', lambda_l2=0.01):
        super().__init__()
        self._name=name
        self.regu =tf.keras.regularizers.L2(l2=lambda_l2)
        self.filtersA =  int(filters*2/3)
        self.filtersA2 =  int(filters*1/2)
        self.filtersB =  int(filters*1/3)
        self.filtersC =  int(filters*1/6)
        self.filtersD =  int(filters*1/12)


        self.a1 = Conv2D(self.filtersD, (1,1), strides=(1, 1), padding="same",
                    activation='relu', kernel_regularizer=self.regu)
        self.a2 = Conv2D(self.filtersD, (3,3), strides=(1, 1), padding="same",
                    activation='relu', kernel_regularizer=self.regu)
        self.a3 = Conv2D(self.filtersC, (3,3), strides=(1, 1), padding="same",
                    activation='relu', kernel_regularizer=self.regu)
        self.b1 = Conv2D(self.filtersA2, (1,1), strides=(1, 1), padding="same",
                    activation='relu', kernel_regularizer=self.regu)
        self.b2 = Conv2D(self.filtersA, (3,3), strides=(1, 1), padding="same",
                    activation='relu', kernel_regularizer=self.regu)
        self.c1 = MaxPooling2D((2,2), strides=(1, 1), padding="same")
        self.c2 = Conv2D(self.filtersC, (1,1), strides=(1, 1), padding="same",
                    activation='relu', kernel_regularizer=self.regu)
        self.d1 = Conv2D(self.filtersB, (1,1), strides=(1, 1), padding="same",
                    activation='relu', kernel_regularizer=self.regu)
        
    def call(self, inputs):
        x1 = self.a1(inputs)
        x1 = self.a2(x1)
        x1 = self.a3(x1)

        x2 = self.b1(inputs)
        x2 = self.b2(x2)

        x3 = self.c1(inputs)
        x3 = self.c2(x3)

        x4 = self.d1(inputs)
        return Concatenate()([x1,x2,x3,x4])
    

class FaceRecog(Model):
    def __init__(self, imgShape, name='FaceRecoginiton', lambda_l2=0.01):
        super().__init__()
        self._name=name
        self.regu =tf.keras.regularizers.L2(l2=lambda_l2)

        self.inp = Input(shape=imgShape)
        self.c1 = Conv2D(64, (10,10), strides=(1, 1), padding="valid",
                         activation='relu', kernel_regularizer=self.regu)
        self.m1 = MaxPooling2D((2,2))

        self.c2 = Conv2D(128, (7,7), strides=(1, 1), padding="same",
                         activation='relu', kernel_regularizer=self.regu)
        self.m2 = MaxPooling2D((2,2))

        self.c3 = Conv2D(128, (5,5), strides=(1, 1), padding="same",
                          activation='relu', kernel_regularizer=self.regu)
        self.m3 = MaxPooling2D((2,2))

        self.incep = InceptionV2(filters=128, lambda_l2=lambda_l2)

        self.c4 = Conv2D(169, (3,3), strides=(1, 1), padding="same",
                         activation='relu', kernel_regularizer=self.regu)
        self.c4b = Conv2D(169, (1,1), strides=(1, 1), padding="same",
                         activation='relu', kernel_regularizer=self.regu)
        self.m4 = MaxPooling2D((2,2))
        self.c5 = Conv2D(256, (3,3), strides=(1, 1), padding="same",
                          activation='relu', kernel_regularizer=self.regu)
        self.c5b = Conv2D(256, (1,1), strides=(1, 1), padding="same",
                          activation='relu', kernel_regularizer=self.regu)
        # self.m5 = MaxPooling2D((2,2))

        self.c6 = Conv2D(512, (3,3), strides=(1, 1), padding="same",
                          activation='relu', kernel_regularizer=self.regu)
        self.c6b = Conv2D(512, (1,1), strides=(1, 1), padding="same",
                          activation='relu', kernel_regularizer=self.regu)
        # self.m6 = MaxPooling2D((2,2))
        self.f1 = Flatten()
        self.fc1 = Dense(2304, kernel_regularizer=self.regu)
        self.fc2 = Dense(2304, activation='sigmoid', kernel_regularizer=self.regu)

    def call(self, inputs):
        x = self.c1(inputs)
        x = self.m1(x)

        x = self.c2(x)
        x = self.m2(x)

        x = self.c3(x)
        x = self.m3(x)

        x = self.incep(x)

        x = self.c4(x)
        x = self.c4b(x)
        # x = self.m4(x)

        x = self.c5(x)
        x = self.c5b(x)
        # x = self.m5(x)

        x = self.c6(x)
        x = self.c6b(x)
        # x = self.m6(x)

        x = self.f1(x)
        x = self.fc1(x)
        return self.fc2(x)
    
    def summary(self):
        model = Model(inputs=[self.inp], outputs=[self.call(self.inp)], name=self.name)
        return model.summary()

# fr = FaceRecog(imgShape=(105,105,3), lambda_l2=0.01)
# fr.summary()

class SiameseFE(Model):
    def __init__(self, imgShape, name='SiameseModel', lambda_l2=0.01):
        super().__init__()
        self._name=name
        self.regu =tf.keras.regularizers.L2(l2=lambda_l2)

        self.inp = Input(shape=imgShape)
        self.c1 = Conv2D(64, (10,10), strides=(1, 1), padding="valid",
                         activation='relu', kernel_regularizer=self.regu)
        self.m1 = MaxPooling2D((2,2))

        self.c2 = Conv2D(128, (7,7), strides=(1, 1), padding="valid",
                          activation='relu', kernel_regularizer=self.regu)
        self.m2 = MaxPooling2D((2,2))

        self.c3 = Conv2D(128, (4,4), strides=(1, 1), padding="valid",
                          activation='relu', kernel_regularizer=self.regu)
        self.m3 = MaxPooling2D((2,2))

        self.c4 = Conv2D(256, (4,4), strides=(1, 1), padding="valid",
                          activation='relu', kernel_regularizer=self.regu)
        self.f1 = Flatten()
        self.fc1 = Dense(4096, activation='sigmoid', kernel_regularizer=self.regu)
        #self.fc2 = Dense(4096, activation='sigmoid', kernel_regularizer=regu)

    def call(self, inputs):
        x = self.c1(inputs)
        x = self.m1(x)

        x = self.c2(x)
        x = self.m2(x)

        x = self.c3(x)
        x = self.m3(x)

        x = self.c4(x)
        x = self.f1(x)
        x = self.fc1(x)
        return x
    
    def summary(self):
        model = Model(inputs=[self.inp], outputs=[self.call(self.inp)], name=self.name)
        return model.summary()

class L1Dist(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__()
        self._name = 'L1Distance'
    def call(self, input_embedding, output_embedding):
        return tf.math.abs(input_embedding - output_embedding)

class L2Dist(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__()
        self._name = 'L2Distance'
    def call(self, anchor_embedding, positive_embedding, negative_embedding):
        return tf.reduce_sum(tf.square(anchor_embedding - positive_embedding),axis=1)- \
                tf.reduce_sum(tf.square(anchor_embedding - negative_embedding),axis=1)

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

def siamese_TL(siamese_conv_model, imgShape, lambda_l2=0.01):
    regu =tf.keras.regularizers.L2(l2=lambda_l2)
    # Creating input shape
    anchor_image = Input(name='Anchor image', shape=imgShape)
    posi_image= Input(name='Positive image', shape=imgShape)
    nega_image= Input(name='Negative image', shape=imgShape)

    # Passing the input to the siamese convolution module
    input1 = siamese_conv_model(anchor_image)
    input2 = siamese_conv_model(posi_image)
    input3 = siamese_conv_model(nega_image)

    # Calculating the L1 distance between the images
    siamese_L2dist = L2Dist()
    distance = siamese_L2dist(input1, input2, input3)

    #Creating a classifier comparing the distance between pictures
    # classifier = Dense(1, activation='relu', kernel_regularizer=regu)(distance)

    return Model(inputs=[anchor_image, posi_image, nega_image], outputs=[distance], name='SiameseNN')

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