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
import cv2 as cv
import numpy as np
from tqdm import tqdm
from itertools import combinations

#%% Adding paths and directories

PATH = dict()
PATH['main'] = 'C:\\Users\\javie\\Python\\TFOD\\siamese_face_recognition'
PATH['download'] = os.path.join(PATH['main'],'dataset\\raw')
PATH['raw'] = os.path.join(PATH['download'],'lfw')
PATH['preprocess'] = PREP_PATH
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
raw_files = list()
n_files = list()
for dir in directories:
    file = os.listdir(os.path.join(PATH['raw'],dir))
    n_files.append(len(file))
    for f in file:
        raw_files.append(os.path.join(PATH['raw'], dir, f))
        files.append(os.path.join(PATH['preprocess'], dir, f))

#Create preprocessed images with just faces and (105,105) pixels

# weights = os.path.join(PATH['main'], "face_detection_yunet_2023mar.onnx")
# face_detector = cv.FaceDetectorYN_create(weights, "", (250,250), score_threshold=0.7)

def face_rc(img, face_detector):
    fr = face_detector.detect(img)
    if fr[0] == 0:
        print('There are no faces here!!!')
    else:
        if fr[1].shape[0] == 1:
            x, y, width, height = face_detector.detect(img)[1][0].astype(np.int16)[:4]
        else:
            better_score = 0
            min_dist = 100
            mid = np.array([125.0,125.0])
            for g, guess in enumerate(fr[1]):
                ar = guess[:4]
                cm = ar[:2] + ar[2:]/2.0

                dis = np.sqrt(np.square(cm - mid).sum())
                if min_dist > dis:
                    better_score = g
                    min_dist = dis
                else:
                    pass

            x, y, width, height = face_detector.detect(img)[1][better_score].astype(np.int16)[:4]
        img_l = int(max(105, width, height))
        x_cm = int(x + width/2.0)
        y_cm = int(y + height/2.0)
        x_1 = int(x_cm - img_l/2.0)
        x_2 = int(x_1 + img_l)
        y_1 = int(y_cm - img_l/2.0)
        y_2 = int(y_1 + img_l)
        if min(x_1,y_1) < 0 or max(x_2,y_2)>250:
            if x_1 < 0:
                x_2 = x_2 - x_1
                x_1 = 0
            if x_2 > 250:
                x_1 = x_1 - x_2 + 250
                x_2 = 250
            if y_1 < 0:
                y_2 = x_2 - y_1
                y_1 = 0
            if y_2 > 250:
                y_1 = y_1 - y_2 + 250
                y_2 = 250
        img = img[y_1:y_2,x_1:x_2]
        img = cv.resize(img,dsize=(105,105))
        return img

def image_preprocessing():
    # Creating face recognition algorithm with openCV 
    weights = os.path.join(PATH['main'], "face_detection_yunet_2023mar.onnx")
    face_detector = cv.FaceDetectorYN_create(weights, "", (250,250), score_threshold=0.7)

    # Read each image and create a face sample in the new directory
    for f, file in enumerate(tqdm(raw_files)):
        if not os.path.exists(os.path.dirname(files[f])):
            os.mkdir(os.path.dirname(files[f]))
        else:
            pass
        img = cv.imread(file)
        img = face_rc(img, face_detector)
        cv.imwrite(files[f], img)
    del face_detector, weights


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

#Triplet loss function dataset
# Format (Anchor, positive, negative)

count = 0
triplets = list()
for n, nf in enumerate(n_files):
    #generating 2 negative samples every time the dir has just 1 picture
    if nf == 2:
        images_n = list(range(count,count+nf))
        ram = random.randint(0, len(files)-1)
        while ram in images_n:
            ram = random.randint(0, len(files)-1)
        triplets.append([files[images_n[0]],files[images_n[1]],files[ram]])
    elif nf > 2:
        images_n = list(range(count,count+nf))
        comb = list(combinations(images_n,2))
        for c, com in enumerate(comb):
            ram = random.randint(0, len(files)-1)
            while ram in images_n:
                ram = random.randint(0, len(files)-1)
            triplets.append([files[com[0]],files[com[1]],files[ram]])
    else:
        pass
    count += nf


#%%  Creating the preprocessing function for the images

@tf.function
def preprocessing(img):
    img = tf.io.read_file(img)
    img = tf.io.decode_jpeg(img, channels=3)
    img = tf.cast(img, dtype=tf.float32)/255.0
    img = tf.image.resize(img, size=INPUT_IMAGE_SIZE[:2])
    return img

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

def dataset_generator_triplets(TRAIN_SIZE= int(len(triplets)*0.7), TEST_SIZE=int(len(triplets)*0.2)):

    ds = triplets.copy()
    random.shuffle(ds)
    ds = tf.data.Dataset.from_tensor_slices(ds)

    ds = ds.map(lambda x: (preprocessing(x[0]), preprocessing(x[1]), preprocessing(x[2])), num_parallel_calls=tf.data.AUTOTUNE)

    train = ds.take(TRAIN_SIZE)
    test = ds.skip(TRAIN_SIZE).take(TEST_SIZE)
    val = ds.skip(TRAIN_SIZE).skip(TEST_SIZE)

    del ds

    return train, test, val
