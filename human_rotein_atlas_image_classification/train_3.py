import os, sys, math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")

import tensorflow.keras as keras

from kaggle_data_3 import data_generator
from uitls import f1,f1_loss,show_history,loss_all
from inception_resnet_model_3 import inception_resnet_model

os.environ['CUDA_VISIBLE_DEVICES']='7'
#parameter
INPUT_SHAPE = [299,299]
N_OUT = 28
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

keras.backend.clear_session()

model = inception_resnet_model(INPUT_SHAPE,N_OUT,test = False)
model.create_model()
model.summary()

#data generator
train_generator = data_generator.create_train(train_dataset_info[train_ids.index], BATCH_SIZE, (299,299,3), type_image = 0,augument=False)
validation_generator = data_generator.create_train(train_dataset_info[test_ids.index], 256, (299,299,3), type_image = 0, augument=False)
model.set_generators(train_generator,validation_generator)

model.inception_resnet_trainable(False)
model.compile_model()
history1 = model.learn(False)
model.save('working/model_not_train.h5')
#todo prediction

#data generator
train_generator = data_generator.create_train(train_dataset_info[train_ids.index], BATCH_SIZE, (299,299,3), type_image = 0, augument=True)
validation_generator = data_generator.create_train(train_dataset_info[test_ids.index], 256, (299,299,3), type_image = 0, augument=True)
model.set_generators(train_generator,validation_generator)

model.inception_resnet_trainable(True)
model.compile_model()
history2 = model.learn(True)
model.save('working/model_train.h5')

#show_history(history1)
#show_history(history2)
