# -*- coding: utf-8 -*-

#================================================================
#
#   File name   : dataset_creator.py
#   Author      : Rigel89
#   Created date: 28/06/23
#   GitHub      : https://github.com/Rigel89/siamese_face_recognition
#   Description : initial dataset creator script
#
#================================================================

#%% Importing libraries

import os
import tensorflow as tf
import random
from siamese.config import *

#%% Adding paths and directories

PATH = dict()
PATH['main'] = 'C:\\Users\\javie\\Python\\TFOD\\siamese_face_recognition'
PATH['download'] = os.path.join(PATH['main'],'dataset\\raw')
PATH['raw'] = os.path.join(PATH['download'],'lfw')
PATH['train'] = os.path.join(PATH['main'],'dataset\\train')
PATH['test'] = os.path.join(PATH['main'],'dataset\\test')

#%%  Download and uncompresse files

#Donwload from: http://vis-www.cs.umass.edu/lfw/#download
#os.system('tar -xf {}'.format(os.path.join(PATH['download'],'lfw.tgz')))

#%%  Getting information about the dataset

#Creating a list of files inside every character and the number of files of each directory.
# data structure:
# dir: raw/lfw
# -> dir: Characters, E.g.:Aaron_Eckhart
# -> -> file: picture, E.g.:Aaron_Eckhart_picture01.png

directories = os.listdir(PATH['raw'])
files = list()
n_files = list()
for dir in directories:
    file = os.listdir(os.path.join(PATH['raw'],dir))
    n_files.append(len(file))
    for f in file:
        files.append(os.path.join(PATH['raw'], dir, f))

#%% Creating the algorithm to get the dataset

# %% [markdown]
# The dataset for Siamese neuronal network:
# <ol>
#     <li>The siamese nn has 2 inputs of 2 pictures and one output, if the 2 pictures correspond to the same person 1 or 0 when they are different people.</li>
#     <li>Positive samples will be two pictures of the same person.</li>
#     <li>Negative samples will be two pictures of different people.</li>
#     <li>The 'true predictions' for the training will be an array with zeros and ones, 0 for negatives and 1 for positive.</li>
# </ol>
# 
# Directives to create an efective algorithm in the dataset:
# <ul>
#     <li>Every folder with just one picture is going to be a negative example.</li>
#     <li>Every negative example will be use twice with two different pictures in different folders to balance the dataset (same number of negative than positive samples).</li>
#     <li>Every negative sample will be peer with two ramdom pictures of the list of pictures.</li>
#     <li>Every folder with more of 1 picture will be use as positive sample</li>
#     <li>The pictures for potive samples in every folder will be peer with the next one in the list of the folder to create the dataset.</li>
#     <li>In case the pictures is the last in the list of the folder will be peer with the first one.</li>
#     <li>In case the folder has just 2 pictures, there will not be repetition.</li>
# </ul>

# files is a list of paths for every picture in order of extraction for every folder
# n_files is a list with the number of picures every folder

# Fix the ramdom number seed to generate always the same dataset
random.seed(13)

# Create the list of variables
positives = list()
negatives = list()
count = 0

for n, nf in enumerate(n_files):
    #generating 2 negative samples every time the dir has just 1 picture
    if nf == 1:
        iter = 0
        while iter <= 1:
            ram = random.randint(0, len(files)-1)
            if ram != count:
                negatives.append([files[count],files[ram],'0.0'])
                iter += 1
    # Generating 1 positive sample every time the dir has 2 pictures
    elif nf == 2:
        positives.append([files[count], files[count+1],'1.0'])
    # Generating N positive sample every time the dir has N pictures, N greater than 2
    else:
        for r in range(nf):
            if r != nf-1:
                positives.append([files[count+r], files[count+r+1],'1.0'])
            else:
                positives.append([files[count+r], files[count],'1.0'])
    count += nf



#%%  Creating the preprocessing function for the images

@tf.function
def preprocessing(x):
    x = tf.io.read_file(x)
    x = tf.io.decode_jpeg(x, channels=3)
    x = tf.cast(x, dtype=tf.float32)/255.0
    x = tf.image.resize(x, size=INPUT_IMAGE_SIZE[:2])
    return x

#%% Creating the dataset extractor function

def dataset_generator(TRAIN_SIZE, TEST_SIZE):

    ds = positives + negatives
    random.shuffle(ds)
    ds = tf.data.Dataset.from_tensor_slices(ds)

    ds = ds.map(lambda x: (preprocessing(x[0]),preprocessing(x[1]), float(x[2])), num_parallel_calls=tf.data.AUTOTUNE)

    train = ds.take(TRAIN_SIZE)
    test = ds.skip(TRAIN_SIZE).take(TEST_SIZE)
    val = ds.skip(TRAIN_SIZE).skip(TEST_SIZE)

    del ds

    return train, test, val

