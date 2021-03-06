import os, sys, math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from imgaug import augmenters as iaa
from tqdm import tqdm


from tensorflow.keras.metrics import categorical_accuracy
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.models import Model
from tensorflow.keras.applications import InceptionResNetV2
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import LambdaCallback
from tensorflow.keras.callbacks import Callback
from tensorflow.keras import metrics
from tensorflow.keras.optimizers import Adam 
from tensorflow.keras import backend as K
from tensorflow import keras
import tensorflow as tf


from sklearn.model_selection import train_test_split

from uitls import f1,f1_loss,show_history,focal_loss
from kaggle_data import data_generator  

import warnings
warnings.filterwarnings("ignore")

#parameter
INPUT_SHAPE = [299,299]
BATCH_SIZE = 8
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

def create_model(pretrain_model,input_shape, n_out):
    input_tensor = Input(shape=input_shape+[4])
    out = Conv2D(3, kernel_size=1, strides=1, padding='same', activation='tanh', kernel_regularizer=regularizers.l2(1e-4))(input_tensor)
    bn = BatchNormalization()(out)
    x = pretrain_model(bn)
    '''
    x = Conv2D(128, kernel_size=(1,1), activation='relu')(x)
    x = Flatten()(x)
    x = Dropout(0.5)(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    output = Dense(n_out, activation='sigmoid')(x)
    '''
    output = Dense(28)(x)
    output = Activation('sigmoid')(output)
    model = Model(input_tensor, output)
    
    return model

keras.backend.clear_session()

pretrain_model = InceptionResNetV2(include_top=False, weights='imagenet', input_shape=INPUT_SHAPE+[3],pooling='avg',classes='avg')

model = create_model(pretrain_model,input_shape=INPUT_SHAPE, n_out=28)

model.summary()

#checkpointer = ModelCheckpoint( 'working/InceptionResNetV2.model', verbose=2, save_best_only=True)

#data generator
train_generator = data_generator.create_train(train_dataset_info[train_ids.index], BATCH_SIZE, (299,299,4), augument=False)
validation_generator = data_generator.create_train(train_dataset_info[test_ids.index], 256, (299,299,4), augument=False)

for layer in pretrain_model.layers:
    layer.trainable = False

model.compile(
    loss=focal_loss,  
    optimizer=Adam(1e-3),
    metrics=['categorical_accuracy', 'binary_accuracy', f1])
'''
#test
history = model.fit_generator(
    train_generator,
    steps_per_epoch=100,
    validation_data=next(validation_generator),
    epochs=2, 
    verbose=1,
    callbacks=[checkpointer])

for x in model.trainable_weights:
    print (x.name)
print ('\n')

for x in model.non_trainable_weights:
    print (x.name)
print ('\n')

'''


history = model.fit_generator(
    train_generator,
    steps_per_epoch=1000,
    validation_data=next(validation_generator),
    epochs=20, 
    verbose=1)
    #callbacks=[])

#show_history(history)

model.save('working/model_not_train.h5')


for layer in pretrain_model.layers:
    layer.trainable = True

model.compile(
    loss=focal_loss,  
    optimizer=Adam(1e-4),
    metrics=['categorical_accuracy', 'binary_accuracy', f1])

history1 = model.fit_generator(
    train_generator,
    steps_per_epoch=1000,
    validation_data=next(validation_generator),
    epochs=20, 
    verbose=1)
    #callbacks=[])

#show_history(history1)

model.save('working/model_train1.h5')


def lr_schedule(epoch):
    lr = 1e-4
    print('learning rate:',lr)
    return lr

lr_scheduler = LearningRateScheduler(lr_schedule)

history2 = model.fit_generator(
    train_generator,
    steps_per_epoch=1000,
    validation_data=next(validation_generator),
    epochs=20, 
    verbose=1,
    callbacks=[lr_scheduler])

#show_history(history2)

model.save('working/model_train2.h5')

show_history(history)
show_history(history1)
show_history(history2)
