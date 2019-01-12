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
from uitls import f1,f1_loss,show_history,loss_all,f1_list
from inception_resnet_model import inception_resnet_model

from absl import app
from absl import flags
import csv
import tqdm

FLAGS = flags.FLAGS
flags.DEFINE_string('gpu', None, 'comma separated list of GPU to use.')
flags.DEFINE_integer('input_div', 4, 'input_div')
flags.DEFINE_integer('input_type', 0, 'input_type')
flags.DEFINE_integer('batch_size', 24, 'batch_size')



def snapshot(argv):
    if FLAGS.gpu: 
        os.environ['CUDA_VISIBLE_DEVICES']=FLAGS.gpu
    #parameter
    INPUT_SHAPE = [299,299]
    N_OUT = 28

    INPUT_DIV = FLAGS.input_div
    BATCH_SIZE = FLAGS.batch_size
    input_type = FLAGS.input_type
    
    label_threshold = 0.3
    submit = pd.read_csv('data/sample_submission.csv')
    imagePath = 'data/test/'


    model = inception_resnet_model(INPUT_SHAPE,N_OUT,div=FLAGS.input_div)
    model.load('working/model_train1547133975.h5')
    names = ['test-09.hdf5','test-24.hdf5','test-47.hdf5','test-80.hdf5','test-134.hdf5','test-192.hdf5']

    label_scores = []
    for name in names: 
        model.load_weight(name)
        label_score = model.snapshot_pre(submit, imagePath, input_type)
        label_scores += [label_score]
        #print (label_score)

    label_score = np.mean(label_scores,axis=0)
    print ('label_score',type(label_score),len(label_score), len(label_score[0]))

    predicted = []
    for i in range(len(label_score)):
        label_predict = np.arange(28)[label_score[i]>=label_threshold]
        str_predict_label = ' '.join(str(l) for l in label_predict)
        predicted.append(str_predict_label)

    submit['Predicted'] = predicted
    submit.to_csv('submission.csv', index=False)


def writeList2CSV(res, csvfile):
    with open(csvfile, "w") as output:
        writer = csv.writer(output)
        for val in res:
            writer.writerow(val)   


if __name__ == '__main__':
    app.run(snapshot)


