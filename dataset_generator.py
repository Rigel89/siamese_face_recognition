# -*- coding: utf-8 -*-

#================================================================
#
#   File name   : dataset_generator.py
#   Author      : Rigel89
#   Created date: 26/06/23
#   GitHub      : https://github.com/Rigel89/siamese_face_recognition
#   Description : dataset creator script
#
#================================================================

#%% Libraries

import os
import tensorflow as tf

#%% Paths

PATH = dict()
PATH['main'] = 'C:\\Users\\javie\\Python\\TFOD\\siamese_face_recognition'
PATH['download'] = os.path.join(PATH['main'],'dataset\\raw')
PATH['train'] = os.path.join(PATH['main'],'dataset\\train')
PATH['test'] = os.path.join(PATH['main'],'dataset\\test')

#%% Download and uncomprese

#Donwload from: http://vis-www.cs.umass.edu/lfw/#download
#os.system('tar -xf {}'.format(os.path.join(PATH['download'],'lfw.tgz')))

# %% Getting information

# %%
