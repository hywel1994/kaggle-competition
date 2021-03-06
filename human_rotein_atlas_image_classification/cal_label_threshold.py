import os, sys, math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")

from tensorflow import keras
import time

from kaggle_data import data_generator
from uitls import f1,f1_loss,show_history,loss_all
from inception_resnet_model import inception_resnet_model

from absl import app
from absl import flags


FLAGS = flags.FLAGS
flags.DEFINE_string('gpu', None, 'comma separated list of GPU to use.')
flags.DEFINE_integer('input_div', 3, 'input_div')
flags.DEFINE_integer('input_type', 0, 'input_type')
flags.DEFINE_integer('batch_size', 24, 'batch_size')



def snapshot():
    if FLAGS.gpu: 
        os.environ['CUDA_VISIBLE_DEVICES']=FLAGS.gpu
    #parameter
    INPUT_SHAPE = [299,299]
    N_OUT = 28

    INPUT_DIV = FLAGS.input_div
    BATCH_SIZE = FLAGS.batch_size
    input_type = FLAGS.input_type
    
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

    train_dataset_info = train_dataset_info[:10]

    keras.backend.clear_session()
    
    model = inception_resnet_model(INPUT_SHAPE,N_OUT,div=3)
    model.load('working/model_train1546768075.h5')
    name = 'working/log_1_6/test2-22.hdf5'
    model.load_weight(name)

    labels_true, scores_predict = model.eval(dataset_info, input_type)

    f1_val = []

    for threshold in np.arange(0,1,0.05):
        labels_predict = np.arange([10,28])[scores_predict>=threshold]

        f1_val += [f1(labels_true,labels_predict)]

    print ('f1_val = ', f1_val)
