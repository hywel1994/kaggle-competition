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
flags.DEFINE_string('tb_log_dir', None, 'tb_log_dir.')
flags.DEFINE_string('scheduler_type', None, 'scheduler_type')
flags.DEFINE_integer('test',0,'test')
flags.DEFINE_integer('augument',0,'augument')


def train(argv):
    if FLAGS.gpu: 
        os.environ['CUDA_VISIBLE_DEVICES']=FLAGS.gpu
    #parameter
    INPUT_SHAPE = [299,299]
    N_OUT = 28

    INPUT_DIV = FLAGS.input_div
    BATCH_SIZE = FLAGS.batch_size
    input_type = FLAGS.input_type

    log_dir = FLAGS.tb_log_dir
    scheduler_type = FLAGS.scheduler_type
    test = FLAGS.test
    augument = FLAGS.augument

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

    model = inception_resnet_model(INPUT_SHAPE,N_OUT,div=INPUT_DIV, scheduler_type=scheduler_type, test = test, log_dir = log_dir)
    model.create_model()
    model.summary()

    #data generator
    train_generator = data_generator.create_train(train_dataset_info[train_ids.index], BATCH_SIZE, (299,299,INPUT_DIV), type_image = input_type,augument=augument)
    validation_generator = data_generator.create_train(train_dataset_info[test_ids.index], 256, (299,299,INPUT_DIV), type_image = input_type, augument=augument)
    model.set_generators(train_generator,validation_generator)

    model.inception_resnet_trainable(False)
    model.compile_model()
    history1 = model.learn(False)
    name = 'working/model_not_train{}.h5'.format(int(time.time()))
    model.save(name)
    #todo prediction

    #data generator
    train_generator = data_generator.create_train(train_dataset_info[train_ids.index], BATCH_SIZE, (299,299,INPUT_DIV), type_image = input_type, augument=augument)
    validation_generator = data_generator.create_train(train_dataset_info[test_ids.index], 256, (299,299,INPUT_DIV), type_image = input_type, augument=augument)
    model.set_generators(train_generator,validation_generator)

    model.inception_resnet_trainable(True)
    model.compile_model()
    history2 = model.learn(True)
    name = 'working/model_train{}.h5'.format(int(time.time()))
    model.save(name)


if __name__ == '__main__':
    app.run(train)