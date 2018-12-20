import os, sys, math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import cv2
from imgaug import augmenters as iaa
from tqdm import tqdm

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

from sklearn.model_selection import train_test_split

from uitls import f1,f1_loss,show_history
from data_generator import data_generator  

import warnings
warnings.filterwarnings("ignore")

#parameter
INPUT_SHAPE = [299,299]
BATCH_SIZE = 32
src_dir = 'data'
path_to_train = 'data/train'
data = pd.read_csv('data/train.csv')

#train and vali data
train_dataset_info = []
for name, labels in zip(data['Id'], data['Target'].str.split(' ')):
    train_dataset_info.append({
        'path':os.path.join(path_to_train, name),
        'labels':np.array([int(label) for label in labels])})
train_dataset_info = np.array(train_dataset_info)

train_ids, test_ids, train_targets, test_target = train_test_split(
    data['Id'], data['Target'], test_size=0.2, random_state=42)

# y_cat_train_dic = {}
# for icat in range(28):
#     target = str(icat)
#     y_cat_train_5 = np.array([int(target in ee.split()) for ee in train_labels.Target.tolist()])
#     y_cat_train_dic[icat] = y_cat_train_5

def create_model(pretrain_model, input_shape, n_out):

    input_tensor = Input(shape=input_shape+[4])
    input_tensor = Conv2D(3, kernel_size=1, strides=1, padding='same', activation='tanh', kernel_regularizer=regularizers.l2(1e-4))(input_tensor)
    bn = BatchNormalization()(input_tensor)
    x = pretrain_model(bn)
    x = Conv2D(128, kernel_size=(1,1), activation='relu')(x)
    x = Flatten()(x)
    x = Dropout(0.5)(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    output = Dense(n_out, activation='sigmoid')(x)
    model = Model(input_tensor, output)
    
    return model

keras.backend.clear_session()
pretrain_model = InceptionResNetV2(include_top=False, weights='imagenet', input_shape=input_shape+[3])

model = create_model(pretrain_model, input_shape=INPUT_SHAPE, n_out=28)

model.summary()

checkpointer = ModelCheckpoint( 'working/InceptionResNetV2.model', verbose=2, save_best_only=True)

#data generator
train_generator = data_generator.create_train(
    train_dataset_info[train_ids.index], BATCH_SIZE, INPUT_SHAPE, augument=False)
validation_generator = data_generator.create_train(
    train_dataset_info[test_ids.index], 256, INPUT_SHAPE, augument=False)

for layer in pretrain_model.layers:
    layer.trainable = False

model.compile(
    loss=f1_loss,  
    optimizer=Adam(1e-3),
    metrics=['categorical_accuracy', 'binary_accuracy', f1])

history = model.fit_generator(
    train_generator,
    steps_per_epoch=100,
    validation_data=next(validation_generator),
    epochs=15, 
    verbose=1,
    callbacks=[checkpointer])

#show_history(history)

#data generator
train_generator = data_generator.create_train(
    train_dataset_info[train_ids.index], BATCH_SIZE, INPUT_SHAPE, augument=True)
validation_generator = data_generator.create_train(
    train_dataset_info[test_ids.index], 256, INPUT_SHAPE, augument=False)

for layer in pretrain_model.layers:
    layer.trainable = True

model.compile(
    loss=f1_loss,  
    optimizer=Adam(1e-4),
    metrics=['categorical_accuracy', 'binary_accuracy', f1])

history = model.fit_generator(
    train_generator,
    steps_per_epoch=100,
    validation_data=next(validation_generator),
    epochs=180, 
    verbose=1,
    callbacks=[checkpointer])

#show_history(history)
