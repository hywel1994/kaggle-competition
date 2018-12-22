from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, load_model
from keras.layers import Activation
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Input
from keras.layers import BatchNormalization
from keras.layers import Conv2D
from keras.models import Model
from keras.applications import InceptionResNetV2
from keras.callbacks import ModelCheckpoint
from keras.callbacks import LambdaCallback
from keras.callbacks import Callback
from keras import metrics
from keras.optimizers import Adam 
from keras import backend as K
import tensorflow as tf
import keras

from sklearn.metrics import f1_score

import os, sys, math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import cv2
from imgaug import augmenters as iaa
from tqdm import tqdm

K_epsilon = K.epsilon()



def f1(y_true, y_pred):
    tp = K.sum(K.cast(y_true*y_pred, 'float'), axis=0)
    fp = K.sum(K.cast((1-y_true)*y_pred, 'float'), axis=0)
    fn = K.sum(K.cast(y_true*(1-y_pred), 'float'), axis=0)

    p = tp / (tp + fp + K.epsilon())
    r = tp / (tp + fn + K.epsilon())

    f1 = 2*p*r / (p+r+K.epsilon())
    f1 = tf.where(tf.is_nan(f1), tf.zeros_like(f1), f1)
    return K.mean(f1)

def f1_loss(y_true, y_pred):
    
    #y_pred = K.cast(K.greater(K.clip(y_pred, 0, 1), THRESHOLD), K.floatx())
    tp = K.sum(K.cast(y_true*y_pred, 'float'), axis=0)
    tn = K.sum(K.cast((1-y_true)*(1-y_pred), 'float'), axis=0)
    fp = K.sum(K.cast((1-y_true)*y_pred, 'float'), axis=0)
    fn = K.sum(K.cast(y_true*(1-y_pred), 'float'), axis=0)

    p = tp / (tp + fp + K_epsilon)
    r = tp / (tp + fn + K_epsilon)

    f1 = 2*p*r / (p+r+K_epsilon)
    f1 = tf.where(tf.is_nan(f1), tf.zeros_like(f1), f1)
    return 1-K.mean(f1)

def show_arr(arr, nrows = 1, ncols = 4, figsize=(15, 5)):
    fig, subs = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
    for ii in range(ncols):
        iplt = subs[ii]
        try:
            img_array = arr[:,:,ii]
            if ii == 0:
                cp = 'Greens'
            elif ii == 1:
                cp = 'Blues'
            elif ii == 2:
                cp = 'Reds'
            else:
                cp = 'Oranges'
            iplt.imshow(img_array, cmap=cp)
        except:
            pass

def get_arr0(Id, src_dir, test=False):
    def fn(Id, color, test=False):
        if test:
            tgt = 'test'
        else:
            tgt = 'train'
        with open(os.path.join(src_dir, tgt, Id+'_{}.png'.format(color)), 'rb') as fp:
            img = Image.open(fp)
            arr = (np.asarray(img) / 255.)
        return arr
    res = []
    for icolor in ['green', 'blue', 'red', 'yellow']:
        arr0 = fn(Id, icolor, test)
        res.append(arr0)
    arr = np.stack(res, axis=-1)
    return arr

def get_arr(Id, src_dir, shape_2, test=False):
    if test:
        arr = get_arr0(Id, src_dir, test=True)
    else:
        arr = get_arr0(Id, src_dir)
    arr = cv2.resize(arr, shape_2).astype('float32')
    return arr

def show_history(history):
    fig, ax = plt.subplots(1, 4, figsize=(15,5))
    ax[0].set_title('loss')
    ax[0].plot(history.epoch, history.history["loss"], label="Train loss")
    ax[0].plot(history.epoch, history.history["val_loss"], label="Validation loss")
    ax[1].set_title('f1')
    ax[1].plot(history.epoch, history.history["f1"], label="Train f1")
    ax[1].plot(history.epoch, history.history["val_f1"], label="Validation f1")
    ax[2].set_title('c_acc')
    ax[2].plot(history.epoch, history.history["categorical_accuracy"], label="Train c_acc")
    ax[2].plot(history.epoch, history.history["val_categorical_accuracy"], label="Validation c_acc")
    ax[3].set_title('b_acc')
    ax[3].plot(history.epoch, history.history["binary_accuracy"], label="Train b_acc")
    ax[3].plot(history.epoch, history.history["val_binary_accuracy"], label="Validation b_acc")
    ax[0].legend()
    ax[1].legend()
    ax[2].legend()
    ax[3].legend()
    plt.show()



# def calc_threshold(pred):
#     ### calc threshold
#     threshold_dic = {}
#     for idx in tqdm(range(28)):
#         m = 0
#         for ii in range(100):
#             threshold0 = ii*0.01
#             f1_val = f1_score(y_cat_train_dic[idx], threshold0<(pred[:,idx]))
#             if m < f1_val:
#                 threshold_dic[idx] = threshold0+0.005
#                 m = f1_val
#     return threshold_dic