import os, sys, math
import numpy as np
import pandas as pd
from PIL import Image
from skimage.transform import resize
from imgaug import augmenters as iaa
from tqdm import tqdm

import warnings
warnings.filterwarnings("ignore")

class data_generator:
    def create_train(dataset_info, batch_size, shape, augument=True):
        assert shape[2] == 4
        while True:
            random_indexes = np.random.choice(len(dataset_info), batch_size)
            batch_images = np.empty((batch_size, shape[0], shape[1], shape[2]))
            batch_labels = np.zeros((batch_size, 28))
            for i, idx in enumerate(random_indexes):
                image = data_generator.load_image(
                    dataset_info[idx]['path'], shape)   
                if augument:
                    image = data_generator.augment(image)
                batch_images[i] = image
                batch_labels[i][dataset_info[idx]['labels']] = 1
            yield batch_images, batch_labels
            
    def create_train_all(dataset_info, shape, augument=True):
        assert shape[2] == 4
        images_all = np.empty((len(dataset_info), shape[0], shape[1], shape[2]))
        labels_all = np.zeros((len(dataset_info), 28))
        for i, idx in enumerate(dataset_info):
            image = data_generator.load_image(
                dataset_info[idx]['path'], shape)   
            if augument:
                image = data_generator.augment(image)
            images_all[i] = image
            labels_all[i][dataset_info[idx]['labels']] = 1
        return images_all, labels_all

    def load_image(path, shape):
        R = np.array(Image.open(path+'_red.png'))
        G = np.array(Image.open(path+'_green.png'))
        B = np.array(Image.open(path+'_blue.png'))
        Y = np.array(Image.open(path+'_yellow.png'))

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
