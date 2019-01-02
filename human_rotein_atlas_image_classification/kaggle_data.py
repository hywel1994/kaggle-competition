import os, sys, math
import numpy as np
import pandas as pd
from PIL import Image
from skimage.transform import resize
from imgaug import augmenters as iaa
#from keras.preprocessing.image import ImageDataGenerator
#from keras.preprocessing import image

import warnings
warnings.filterwarnings("ignore")

class data_generator:
    def create_train(dataset_info, batch_size, shape, type_image=0,augument=True):
        #assert shape[2] == 3
        while True:
            random_indexes = np.random.choice(len(dataset_info), batch_size)
            batch_images = np.empty((batch_size, shape[0], shape[1], shape[2]))
            batch_labels = np.zeros((batch_size, 28))
            for i, idx in enumerate(random_indexes):
                image = data_generator.load_image(
                    dataset_info[idx]['path'], shape, type_image)   
                if augument:
                    image = data_generator.augment(image)
                batch_images[i] = image
                batch_labels[i][dataset_info[idx]['labels']] = 1
            yield batch_images, batch_labels

    def load_image(path, shape, type_image):
        
        R = np.array(Image.open(path+'_red.png'))    # microtubules
        G = np.array(Image.open(path+'_green.png'))  # main protein
        B = np.array(Image.open(path+'_blue.png'))   # nucleus
        Y = np.array(Image.open(path+'_yellow.png')) # reticulum
        if shape[2]==3:
            if type_image==0:
                image = np.stack((
                    G, 
                    R, 
                    Y),-1)
            elif type_image==1:
                image = np.stack((
                    G, 
                    B, 
                    Y),-1)
            elif type_image==2:
                image = np.stack((
                    G, 
                    B, 
                    R),-1) 
            elif type_image==3:
                image = np.stack((
                    G, 
                    R*2/3+B/3, 
                    Y*2/3+B/3),-1) 
        
        elif shape[2]==4:
            image = np.stack((
            G, 
            R, 
            B,
            Y),-1)

        image = resize(image, (shape[0], shape[1]))
        image = np.divide(image, 255)
        return image
                
    def augment(image):
        augment_img = iaa.Sequential([
            iaa.OneOf([
                iaa.Affine(rotate=0),
                iaa.Affine(rotate=90),
                iaa.Affine(rotate=180),
                iaa.Affine(rotate=270),
                iaa.Fliplr(0.5),
                iaa.Flipud(0.5),
            ])], random_order=True)
        
        image_aug = augment_img.augment_image(image)
        return image_aug
