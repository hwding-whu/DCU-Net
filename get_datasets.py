import numpy as np
import random
import configparser

from help_functions import load_hdf5
from help_functions import visualize
from help_functions import group_images

from pre_processing import my_PreProc


#To select the same images
# random.seed(10)

#Load the original data and return the extracted patches for training/testing
#get_data_training(原始眼底图片地址，分割眼底图片地址，高度，宽度，分割数)
def get_data_training(train_imgs_original,train_groudTruth):
    train_imgs = load_hdf5(train_imgs_original)#加载original数组的np类型的数据
    train_ground = load_hdf5(train_groudTruth) #加载groundTruth数组的np类型数据

    train_imgs = train_imgs/255
    train_ground = train_ground/255
    return train_imgs, train_ground#, patches_imgs_test, patches_masks_test


#Load the original data and return the extracted patches for training/testing
def get_data_testing(test_imgs_original, test_groudTruth):
    ### test
    test_imgs = load_hdf5(test_imgs_original)
    test_ground = load_hdf5(test_groudTruth)

    test_imgs = test_imgs/255
    test_ground = test_ground/255
    return test_imgs, test_ground

'''
def get_data_training0(train_imgs_original0,train_groudTruth0):
    train_imgs0 = load_hdf5(train_imgs_original0)#加载original数组的np类型的数据
    train_ground0 = load_hdf5(train_groudTruth0) #加载groundTruth数组的np类型数据

    train_imgs0 = train_imgs0/255
    train_ground0 = train_ground0/255
    return train_imgs0, train_ground0#, patches_imgs_test, patches_masks_test


#Load the original data and return the extracted patches for training/testing
def get_data_testing0(test_imgs_original0, test_groudTruth0):
    ### test
    test_imgs0 = load_hdf5(test_imgs_original0)
    test_ground0 = load_hdf5(test_groudTruth0)

    test_imgs0 = test_imgs0/255
    test_ground0 = test_ground0/255
    return test_imgs0, test_ground0
'''