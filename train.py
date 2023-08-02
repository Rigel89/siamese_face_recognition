# -*- coding: utf-8 -*-

#================================================================
#
#   File name   : train.py
#   Author      : Rigel89
#   Created date: 28/06/23
#   GitHub      : https://github.com/Rigel89/siamese_face_recognition
#   Description : training script
#
#================================================================

#%% Importing libraries

import tensorflow as tf
from tensorflow.keras.metrics import BinaryAccuracy, Recall, Precision
import os
import numpy as np
from shutil import rmtree


#%% Set the gpu

#This code must be here because has to be set before made other operations (quit ugly solution!!)
print('SETTING UP GPUs')
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
            print('Setting up the GPUs done!')
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print('Setting up the GPUs Not done')


#%% Import other libraries (this include modules and functions using tf)

from siamese.config import *
from siamese.architecture import *
from siamese.lossFunction import *
from dataset_creator import dataset_generator

#%% Define de main function and the training steps

#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' #disable tf info and warning during training

# 0 = all messages are logged (default behavior)
# 1 = INFO messages are not printed
# 2 = INFO and WARNING messages are not printed
# 3 = INFO, WARNING, and ERROR messages are not printed

def main():
    global TRAIN_FROM_CHECKPOINT
    
    # Delete existing path and write a new log file
    print('1.Checking the existing path to the log:')
    if os.path.exists(os.path.join(MAIN_PATH,TRAIN_LOGDIR)):
        print('    There are existing log and events files...')
        print('    KEEP_LOG_EVENTS is set to: ', KEEP_LOG_EVENTS)
        if KEEP_LOG_EVENTS:
            print('        *The files will be kept')
        else:
            rmtree(os.path.join(MAIN_PATH,TRAIN_LOGDIR))
            print('        *Deleting files!!')
    else:
        print('    No existing path')
    print()

    print('2.Creating new log file')
    print()
    writer = tf.summary.create_file_writer(os.path.join(MAIN_PATH,TRAIN_LOGDIR))

    # Load the dataset
    print('3.Loading training dataset:')
    trainset, testset, valset = dataset_generator(TRAIN_SIZE, TEST_SIZE)
    print('    Done!')
    print()
    print('4.Dataset information')
    print('    The train dataset has ', TRAIN_SIZE, 'samples')
    print('    The test dataset has ', TEST_SIZE, 'samples')
    print('    The validation dataset has ', VAL_SIZE, 'samples')
    print()

    # Set the configuration parameters for the dataset during training

    trainset = trainset.shuffle(TRAIN_SHUFFLE).batch(TRAIN_BATCH).prefetch(tf.data.AUTOTUNE)
    testset = testset.batch(TRAIN_BATCH).prefetch(tf.data.AUTOTUNE)
    valset = valset.batch(TRAIN_BATCH).prefetch(tf.data.AUTOTUNE)

    print('5.Training dataset is configured:')
    print('    Batch:               ', TRAIN_BATCH)
    print('    Shuffle batch:       ', TRAIN_SHUFFLE)
    print()

    # Training variables for steps
    steps_per_epoch = len(trainset)
    global_steps = tf.Variable(0, trainable=False, dtype=tf.int64)
    warmup_steps = TRAIN_WARMUP_EPOCHS * steps_per_epoch
    total_steps = TRAIN_EPOCHS * steps_per_epoch
    learning_rate_decay = (TRAIN_LR_END/TRAIN_LR_INIT)**(1/(LR_VARIATION_EPOCH*steps_per_epoch))
    print('    Epochs:              ', TRAIN_EPOCHS)
    print('    Steps per epoch:     ', steps_per_epoch)
    print('    Warmup steps:        ', warmup_steps)
    print('    Total steps:         ', total_steps)
    print('    Learning rate decay: ', learning_rate_decay)
    print()

    # Creating the neuronal network
    print('7.Creating neuronal network')
    lambda_regularization = tf.Variable(initial_value=0.0, trainable=False)
    siamese_conv = SiameseFE(INPUT_IMAGE_SIZE, lambda_l2=float(lambda_regularization.numpy()))
    print('-Creating convolutional model:')
    print()
    print('-Convolution summary:')
    siamese_conv.summary()
    print()

    siamese_L1 = L1Dist()
    print('-Creating L1 distance model...')
    print()

    print('-Creating full siamese model:')
    siameseNN = siamese_FM(siamese_conv, INPUT_IMAGE_SIZE, lambda_l2=float(lambda_regularization.numpy()))
    print()
    print('-Convolution summary:')
    siameseNN.summary()
    print()

    # Checking if there is checkpoints to continue training 
    print('8.Training from checkpoint: ' + str(TRAIN_FROM_CHECKPOINT))
    if TRAIN_FROM_CHECKPOINT:
        print("Trying to load weights from check points:")
        try:
            if os.path.exists("./checkpoints"):
                siameseNN.load_weights(f"./checkpoints/{TRAIN_MODEL_NAME}")
                print('    Succesfully load!')
            else:
                print('    There is not existing checkpoint path!')
        except ValueError:
            print("    Shapes are incompatible or there is not checkpoints")
            TRAIN_FROM_CHECKPOINT = False
    
    # Optimizer
    print()
    print('Setting up optimizer Adam and iniciallicating training')
    optimizer = tf.keras.optimizers.Adam()

    # Definin metrics

    BA = BinaryAccuracy(threshold=0.5)
    Rc = Recall(thresholds=0.5)
    Pc = Precision(thresholds=0.5)

    # Defining training step
    
    alpha = 1.0

    def train_step( X, y):
        with tf.GradientTape() as tape:
            pred_result = siameseNN(X, training=True) # There is BatchNormalization layers, so training=True to train the parameters
            loss = 0

            # optimizing process
            loss = alpha*loss_BCE(y, pred_result)

            gradients = tape.gradient(loss, siameseNN.trainable_variables)
            optimizer.apply_gradients(zip(gradients, siameseNN.trainable_variables))

            # Metrics

            BA.update_state(y, pred_result)
            Rc.update_state(y, pred_result)
            Pc.update_state(y, pred_result)

            # writing summary data
            with writer.as_default():
                tf.summary.scalar("lr", optimizer.lr, step=global_steps)
                tf.summary.scalar("l2", lambda_regularization, step=global_steps)
                tf.summary.scalar("loss", loss, step=global_steps)
                tf.summary.scalar("BinaryAccuracy", BA.result().numpy(), step=global_steps)
                tf.summary.scalar("Recall", Rc.result().numpy(), step=global_steps)
                tf.summary.scalar("Precision", Pc.result().numpy(), step=global_steps)
            writer.flush()
            global_steps.assign_add(1)
        return global_steps.numpy(), optimizer.lr.numpy(), lambda_regularization.numpy(),\
            loss.numpy(), BA.result().numpy(), Rc.result().numpy(), Pc.result().numpy()

    # Defining test step

    validate_writer = tf.summary.create_file_writer(os.path.join(MAIN_PATH,TRAIN_LOGDIR))
    def validate_step(X, y):
        pred_result = siameseNN(X, training=False) # There is BatchNormalization layers, so training=False during prediction
        loss=0

        # Metrics

        BA.update_state(y, pred_result)
        Rc.update_state(y, pred_result)
        Pc.update_state(y, pred_result)

        # optimizing process
        loss = alpha*loss_BCE(y, pred_result)
            
        return loss.numpy(), BA.result().numpy(), Rc.result().numpy(), Pc.result().numpy()

    print()
    print('Starting training process:')
    best_binaryaccu = 0 # should be large at start, if it is too small never save the weights
    for epoch in range(TRAIN_EPOCHS):
        for batch in trainset:
            X = batch[:2]
            y = batch[2]
            results = train_step(X,y)
            if results[0]%(steps_per_epoch) == 0:
                cur_step = steps_per_epoch
            else:
                cur_step = results[0]%(steps_per_epoch)
            print("epoch:{:2.0f} step:{:5.0f}/{}, lr:{:.6f}, l2:{:4.4f}, loss:{:4.4f}, BinaryAccuracy:{:4.4f}, Recall:{:4.4f}, Precision:{:4.4f}"
                  .format(epoch, cur_step, steps_per_epoch, results[1], results[2], results[3], results[4], results[5], results[6]), end='\r')
            
            # Update learning rate
            # about warmup: https://arxiv.org/pdf/1812.01187.pdf&usg=ALkJrhglKOPDjNt6SHGbphTHyMcT0cuMJg
            
            if global_steps < warmup_steps:# and not TRAIN_TRANSFER:
                lr = global_steps / warmup_steps * TRAIN_LR_INIT
                optimizer.lr.assign(lr.numpy())
            else:
                # lr = learning_rate_decay*(optimizer.lr.numpy())
                lr = TRAIN_LR_END + 0.5 * (TRAIN_LR_INIT - TRAIN_LR_END)*(
                    (1 + tf.cos((global_steps - warmup_steps) / (total_steps - warmup_steps) * np.pi)))
                if lr > TRAIN_LR_END:
                    # optimizer.lr.assign(lr)
                    optimizer.lr.assign(lr.numpy())
                else:
                    pass
            
            # Update regularization parameter
            if global_steps <= REGULARIZATION_START_EPOCH*steps_per_epoch:
                pass
            else:
                if global_steps <= REGULARIZATION_END_EPOCH*steps_per_epoch:
                    regu = REGULARIZATION_MAX_VALUE/ steps_per_epoch / (REGULARIZATION_END_EPOCH-REGULARIZATION_START_EPOCH) * ( global_steps.numpy() - REGULARIZATION_START_EPOCH*steps_per_epoch)
                    lambda_regularization.assign(regu)
                else:
                    pass
        print("epoch:{:2.0f} step:{:5.0f}/{}, lr:{:.6f}, l2:{:4.4f}, loss:{:4.4f}, BinaryAccuracy:{:4.4f}, Recall:{:4.4f}, Precision:{:4.4f}"
                .format(epoch, cur_step, steps_per_epoch, results[1], results[2], results[3], results[4], results[5], results[6]))
        
        if len(testset) == 0:
            print("configure TEST options to validate model")
            #yolo.save_weights(os.path.join(TRAIN_CHECKPOINTS_FOLDER, TRAIN_MODEL_NAME))
            #continue
        else:
            count, loss_val, BAccu, Rec, Prec = 0, 0, 0, 0, 0
            for batch in testset:
                X = batch[:2]
                y = batch[2]
                results = validate_step(X, y)
                count += 1
                loss_val += results[0]
                BAccu += results[1]
                Rec += results[2]
                Prec += results[3]
            # writing validate summary data
            with validate_writer.as_default():
                tf.summary.scalar("validate_loss", loss_val/count, step=epoch)
                tf.summary.scalar("validate_loss", BAccu/count, step=epoch)
                tf.summary.scalar("validate_loss", Rec/count, step=epoch)
                tf.summary.scalar("validate_loss", Prec/count, step=epoch)
            validate_writer.flush()
            
            print("epoch:{:2.0f} Validation step-> val_loss:{:7.4f}, BinaryAccuracy:{:4.4f}, Recall:{:4.4f}, Precision:{:4.4f}"
                  .format(epoch, loss_val/count, BAccu/count, Rec/count, Prec/count))

        if TRAIN_SAVE_CHECKPOINT and not TRAIN_SAVE_BEST_ONLY:
            save_directory = os.path.join(TRAIN_CHECKPOINTS_FOLDER, TRAIN_MODEL_NAME+"_BAccu_{:7.2f}".format(BAccu/count))
            siameseNN.save_weights(save_directory)
            print('\nWeights saved every epoch\n')
        if TRAIN_SAVE_BEST_ONLY and best_binaryaccu<BAccu/count:
            save_directory = os.path.join(TRAIN_CHECKPOINTS_FOLDER, TRAIN_MODEL_NAME)
            siameseNN.save_weights(save_directory)
            best_binaryaccu = BAccu/count
            print('\nThe weights are being saved this epoch!\n')


if __name__ == '__main__':
    main()