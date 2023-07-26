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

from tensorflow.keras.losses import BinaryCrossentropy

#%% Declaring loss function

loss = BinaryCrossentropy()