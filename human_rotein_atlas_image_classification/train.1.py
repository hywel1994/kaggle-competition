import os, sys, math
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use('TkAgg')

from kaggle_data import data_generator

import autokeras as ak


#parameter
INPUT_SHAPE = [299,299]
N_OUT = 28
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

#data generator
train_images, train_labels = data_generator.create_train_all(train_dataset_info[train_ids.index], (299,299,4), augument=False)
validation_images, validation_labels = data_generator.create_train_all(train_dataset_info[test_ids.index], (299,299,4), augument=False)

#class autokeras_model
clf = ak.ImageClassifier()
clf.fit(train_images, train_labels, time_limit=12 * 60 * 60)
y1 = clf.evaluate(validation_images, validation_labels)
print ('y1 = ',y1)
clf.final_fit(train_images, train_labels, validation_images, validation_labels, retrain=True)
y2 = clf.evaluate(validation_images, validation_labels)
print ('y1 = ',y1, ', y2 = ',y2)
results = clf.predict(validation_images)