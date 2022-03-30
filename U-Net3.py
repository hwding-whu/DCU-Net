# -*- coding: utf-8 -*-
"""
Created on Tue Aug  4 20:03:01 2020

@author: DHW
"""
from typing import Union

import numpy as np
import configparser
import os
import tensorflow as tf
from keras.models import Model
from keras.layers import Input, concatenate,BatchNormalization, Conv2D, Cropping2D, MaxPooling2D,Conv2DTranspose, UpSampling2D,ZeroPadding2D, Reshape, core, Dropout,Add,Activation
from keras.optimizers import Adam,Nadam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as K
from keras.losses import binary_crossentropy
from keras.utils.vis_utils import plot_model
from keras.utils.vis_utils import plot_model as plot
from keras.optimizers import SGD
from keras.applications import vgg16
from get_datasets import get_data_training,get_data_testing
#from get_datasets import get_data_training0
import matplotlib.pyplot as plt
from tensorflow.keras import optimizers
import tensorflow as tf
'''
try:
    from tensorflow.contrib import keras as keras

    print('load keras from tensorflow package')
except:
    print('update your tensorflow')
from tensorflow.contrib.keras import models
from tensorflow.contrib.keras import layers
'''
os.environ["HDF5_USE_FILE_LOCKING"] = 'FALSE'
def get_crop_shape(target, refer):
    # width, the 3rd dimension
    cw = (target.get_shape()[2] - refer.get_shape()[2]).value
    assert (cw >= 0)
    if cw % 2 != 0:
        cw1, cw2 = int(cw / 2), int(cw / 2) + 1
    else:
        cw1, cw2 = int(cw / 2), int(cw / 2)
    # height, the 2nd dimension
    ch = (target.get_shape()[1] - refer.get_shape()[1]).value
    assert (ch >= 0)
    if ch % 2 != 0:
        ch1, ch2 = int(ch / 2), int(ch / 2) + 1
    else:
        ch1, ch2 = int(ch / 2), int(ch / 2)

    return (ch1, ch2), (cw1, cw2)

smooth = 1
def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f*y_true_f) + K.sum(y_pred_f*y_pred_f) + smooth)

def dice_coef_loss(y_true, y_pred):
    return 1.-dice_coef(y_true, y_pred)
'''
def dice_coef(y_true, y_pred, smooth=1):
    intersection = K.sum(y_true * y_pred, axis=[1,2,3])  
    union = K.sum(y_true, axis=[1,2,3]) + K.sum(y_pred, axis=[1,2,3])
    return K.mean( (2. * intersection + smooth) / (union + smooth), axis=0)
'''
def dice_p_bce(in_gt, in_pred):
    return 1e-2*binary_crossentropy(in_gt, in_pred) + dice_coef_loss(in_gt, in_pred)

def Unet(inp1):
    concat_axis = 3
    # input
    #inputs = Input(shape=(in_h,in_w,in_c))
    conv1 = Conv2D(64, (3, 3), activation='relu', padding='same')(inp1)
    conv1 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv1)
    pooling1 = MaxPooling2D((2, 2), strides=(2, 2))(conv1)

    # Block 2
    conv2 = Conv2D(128, (3, 3), activation='relu', padding='same')(pooling1)
    conv2 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv2)
    pooling2 = MaxPooling2D((2, 2), strides=(2, 2))(conv2)

    # Block 3
    conv3 = Conv2D(256, (3, 3), activation='relu', padding='same')(pooling2)
    conv3 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv3)
    conv3 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv3)
    pooling3 = MaxPooling2D((2, 2), strides=(2, 2))(conv3)

    # Block 4
    conv4 = Conv2D(512, (3, 3), activation='relu', padding='same')(pooling3)
    conv4 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv4)
    conv4 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv4)
    pooling4 = MaxPooling2D((2, 2), strides=(2, 2))(conv4)

    # Block 5
    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(pooling4)
    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv5)
    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv5)
    pooling5 = MaxPooling2D((2, 2), strides=(2, 2))(conv5)

    model = Model(inputs=inp1, outputs=[pooling1, pooling2, pooling3, pooling4, pooling5])
    return model

def Unet0(inp2):
    concat_axis = 3
    # input
    #inputs = Input(shape=(in_h, in_w, in_c))

    conv1 = Conv2D(8, (3, 3), activation='relu', padding='same')(inp2)
    conv1 = Conv2D(8, (3, 3), activation='relu', padding='same')(conv1)
    pooling1 = MaxPooling2D((2, 2), strides=(2, 2))(conv1)

    # Block 2
    conv2 = Conv2D(16, (3, 3), activation='relu', padding='same')(pooling1)
    conv2 = Conv2D(16, (3, 3), activation='relu', padding='same')(conv2)
    pooling2 = MaxPooling2D((2, 2), strides=(2, 2))(conv2)

    # Block 3
    conv3 = Conv2D(32, (3, 3), activation='relu', padding='same')(pooling2)
    conv3 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv3)
    conv3 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv3)
    pooling3 = MaxPooling2D((2, 2), strides=(2, 2))(conv3)

    # Block 4
    conv4 = Conv2D(64, (3, 3), activation='relu', padding='same')(pooling3)
    conv4 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv4)
    conv4 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv4)
    pooling4 = MaxPooling2D((2, 2), strides=(2, 2))(conv4)

    # Block 5
    conv5 = Conv2D(128, (3, 3), activation='relu', padding='same')(pooling4)
    conv5 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv5)
    conv5 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv5)
    pooling5 = MaxPooling2D((2, 2), strides=(2, 2))(conv5)

    model = Model(inputs=inp2, outputs=[pooling1, pooling2, pooling3, pooling4, pooling5])
    return model

def merge_model():
    inp1 = Input(shape=(568, 757, 3))
    inp2 = Input(shape=(568, 757, 3))
    model_1 = Unet(inp1)
    model_2 = Unet0(inp2)
    model_1.load_weights(r'./test3/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5')
    model_2.load_weights(r'./test1/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5')

    m1_pool1,m1_pool2,m1_pool3,m1_pool4,m1_pool5 = model_1.output
    m2_pool1,m2_pool2,m2_pool3,m2_pool4,m2_pool5 = model_2.output

    x = concatenate([m1_pool5, m2_pool5], axis=-1)

    x1 = Conv2D(256, (3, 3), activation='relu', padding='same', dilation_rate=1)(x)
    x2 = Conv2D(256, (3, 3), activation='relu', padding='same', dilation_rate=2)(x)
    x3 = Conv2D(256, (3, 3), activation='relu', padding='same', dilation_rate=3)(x)
    x = concatenate([x1, x2, x3], axis=-1)

    x = UpSampling2D(size=(2, 2))(x)
    ch, cw = get_crop_shape(m1_pool4, x)
    m1_pool4 = Cropping2D(cropping=(ch, cw))(m1_pool4)
    x = concatenate([x, m1_pool4], axis=-1)
    x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
    #x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)

    x = UpSampling2D(size=(2, 2))(x)
    ch, cw = get_crop_shape(m1_pool3, x)
    m1_pool3 = Cropping2D(cropping=(ch, cw))(m1_pool3)
    x = concatenate([x, m1_pool3], axis=-1)
    x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    #x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)

    x = UpSampling2D(size=(2, 2))(x)
    ch, cw = get_crop_shape(m1_pool2, x)
    m1_pool2 = Cropping2D(cropping=(ch, cw))(m1_pool2)
    x = concatenate([x, m1_pool2], axis=-1)
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    #x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)

    x = UpSampling2D(size=(2, 2))(x)
    ch, cw = get_crop_shape(m1_pool1, x)
    m1_pool1 = Cropping2D(cropping=(ch, cw))(m1_pool1)
    x = concatenate([x, m1_pool1], axis=-1)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    #x = Conv2D(32, (3, 3), padding='same')(x)

    x = UpSampling2D(size=(2, 2))(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)

    ch, cw = get_crop_shape(inp1, x)
    x = ZeroPadding2D(padding=((ch[0], ch[1]), (cw[0], cw[1])))(x)
    x = Conv2D(1, (1, 1))(x)
    x = BatchNormalization()(x)
    x = Activation("sigmoid")(x)

    model = Model(inputs=[inp1, inp2], outputs=x)
    model.compile(optimizer=Adam(lr=1e-5), loss=dice_p_bce, metrics=['accuracy'])
    return model

config = configparser.RawConfigParser()
config.read('configuration.txt')
#patch to the datasets
path_data = config.get('data paths', 'path_local')
path_data0 = config.get('data paths', 'path_local0')
#Experiment name
name_experiment = config.get('experiment name', 'name3')
N_epochs = int(config.get('training settings', 'N_epochs'))
batch_size = int(config.get('training settings', 'batch_size'))
path_experiment = './' +name_experiment +'/'
best_last = config.get('testing settings', 'best_last')

imgs_train, train_ground = get_data_training(train_imgs_original=path_data + config.get('data paths', 'train_imgs_original'),
                                           train_groudTruth=path_data + config.get('data paths', 'train_groundTruth'))
imgs_train0, imgs_ground0 = get_data_training(train_imgs_original=path_data0 + config.get('data paths', 'train_imgs_original0'),
                                           train_groudTruth=path_data0 + config.get('data paths', 'train_groundTruth0'))

model = merge_model()
model.summary()
plot_model(model, to_file='model_training1.jpg', show_shapes=True)
yaml_string = model.to_yaml()  # type: Union[str, bytes]
with open('./'+name_experiment+'/'+name_experiment +'_architecture.yaml', 'w') as f:
    f.write(yaml_string)
checkpointer = ModelCheckpoint(filepath='./'+name_experiment+'/'+name_experiment +'_best_weights.h5', verbose=1, monitor='val_loss', mode='auto', save_best_only=True) #save at each epoch if the validation decreased
#model.load_weights(path_experiment+name_experiment + '_'+best_last+'_weights.h5')

History=model.fit([imgs_train,imgs_train0],train_ground,epochs=N_epochs, validation_split=0.2, batch_size=batch_size, verbose=1, shuffle=True, callbacks=[checkpointer])

model.save_weights('./'+name_experiment+'/'+name_experiment +'_last_weights.h5', overwrite=True)
