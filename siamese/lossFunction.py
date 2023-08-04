# -*- coding: utf-8 -*-

#================================================================
#
#   File name   : lossFunction.py
#   Author      : Rigel89
#   Created date: 26/06/23
#   GitHub      : https://github.com/Rigel89/siamese_face_recognition
#   Description : loss function script
#
#================================================================

#%% Importing libraries

import tensorflow as tf
from tensorflow.keras.losses import BinaryCrossentropy

#%% Declaring loss function

loss_BCE = BinaryCrossentropy()

# def contrastative_loss(y_true, dist, margin=1000.0):
#     positive = 0.5*tf.multiply(y_true,tf.reduce_sum(dist,axis=1))
#     negative = 0.5*tf.multiply(tf.subtract(1.0,y_true),tf.maximum(0.0,margin-tf.reduce_sum(dist,axis=1)))
#     return tf.reduce_sum(negative + positive), positive, negative 

def contrastative_loss(y_true, dist, margin=10.0):
    positive = 0.5*tf.multiply(y_true,dist)
    negative = 0.5*tf.multiply(tf.subtract(1.0,y_true),tf.maximum(0.0,margin-dist))
    return tf.reduce_sum(negative + positive)

def triplet_loss(dist, margin=15.0):
    return tf.reduce_mean(tf.maximum(0,dist+margin))