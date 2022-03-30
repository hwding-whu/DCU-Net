#==========================================================
#
#  This prepare the hdf5 datasets of the DRIVE database
#
#============================================================

import os
import h5py
import numpy as np
from PIL import Image


os.environ["HDF5_USE_FILE_LOCKING"] = 'FALSE'
def write_hdf5(arr,outfile):
  with h5py.File(outfile,"w") as f:
    f.create_dataset("image", data=arr, dtype=arr.dtype)


#------------Path of the images --------------------------------------------------------------
#train
original_imgs_train = "./Data0/training/images/"
groundTruth_imgs_train = "./Data0/training/groundTruth/"

#test
original_imgs_test = "./Data0/test/images/"
groundTruth_imgs_test = "./Data0/test/groundTruth/"

#---------------------------------------------------------------------------------------------

Nimgs_train = 340
Nimgs_test = 20

channels = 3
height = 568
width = 757
dataset_path = "./datasets_training_testing0/"

def get_datasets(imgs_dir,groundTruth_dir,train_test):
#----------------将original、ground_truth和masks图像转换为np数组-------------------------------------
    #np.empty生成响应的随机数数组
    if (train_test=='train'):
        Nimgs=Nimgs_train
    else:
        Nimgs=Nimgs_test
    
    imgs = np.empty((Nimgs,height,width,channels))
    groundTruth = np.empty((Nimgs,height,width))
    
    for path, subdirs, files in os.walk(imgs_dir): #path是完整地址；files是当前地址下的所有图片
        for i in range(len(files)):
            #original
            print("original image: " +files[i])
            img = Image.open(imgs_dir+files[i])#读取图片，files[0]是21_training.tif
            imgs[i] = np.asarray(img)#数组形状imgs（20,584,565,3）
            #corresponding ground truth
            groundTruth_name = files[i][0:4] + ".png"#files[0]是21_training.tif，files[0][0:2]是取前两个数21
            print("ground truth name: " + groundTruth_name)
            g_truth = Image.open(groundTruth_dir + groundTruth_name)
            groundTruth[i] = np.asarray(g_truth)
           
#--------------------------------------------------------------------------------------------------

    print("imgs max: " +str(np.max(imgs)))#original转为数组后，数组中的最大值(255)
    print("imgs min: " +str(np.min(imgs)))#original转为数组后，数组中的最小值(0)
    assert(np.max(groundTruth)==255)
    #assert(np.min(groundTruth)==0 and np.min(border_masks)==0)
    print("ground truth and border masks are correctly withih pixel value range 0-255 (black-white)")
    #reshaping for my standard tensors
    #imgs = np.transpose(imgs,(0,3,1,2))#original数组形状由(20, 584, 565, 3)转换为(20, 3, 584, 565)
    assert(imgs.shape == (Nimgs,height,width,channels))
    groundTruth = np.reshape(groundTruth,(Nimgs,height,width,1))#groundTruth数组形状转换为（20,1,584,565）  
    assert(groundTruth.shape == (Nimgs,height,width,1))
    return imgs, groundTruth

if not os.path.exists(dataset_path):
    os.makedirs(dataset_path)

#getting the training datasets
imgs_train, groundTruth_train = get_datasets(original_imgs_train,groundTruth_imgs_train,"train")#得到转换为np数组的数据
print("saving train datasets")
write_hdf5(imgs_train, dataset_path+"dataset_imgs_train0.hdf5")#np数组数据存储为.hdf5文件
write_hdf5(groundTruth_train, dataset_path+ "dataset_groundTruth_train0.hdf5")


#getting the testing datasets
imgs_test, groundTruth_test = get_datasets(original_imgs_test,groundTruth_imgs_test,"test")
print("saving test datasets")
write_hdf5(imgs_test, dataset_path +"dataset_imgs_test0.hdf5")
write_hdf5(groundTruth_test, dataset_path+"dataset_groundTruth_test0.hdf5")

