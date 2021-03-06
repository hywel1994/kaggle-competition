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
    
    model = inception_resnet_model(INPUT_SHAPE,N_OUT,div=FLAGS.input_div)
    model.load('working/model_train1547133975.h5')
    names = ['test-09.hdf5','test-24.hdf5','test-47.hdf5','test-80.hdf5','test-134.hdf5','test-192.hdf5']

    label_true = model.snapshot_true(train_dataset_info)
    print ("label_true",type(label_true),len(label_true), len(label_true[0]))
    writeList2CSV(label_true, 'working/label_true.csv')

    label_scores = []
    for name in names: 
        model.load_weight(name)
        label_score = model.snapshot_val(train_dataset_info,input_type)
        #print ('label_score',type(label_score),len(label_score), len(label_score[0]))
        label_scores += [label_score]
        name_flie = 'working/'+name.split('.')[0] + '_label_score.csv'
        writeList2CSV(label_score, name_flie)
    #print ('label_score',label_score)
    #label_scores += [label_score]

    

    label_score = np.mean(label_scores,axis=0)
    print ('label_score',type(label_score),len(label_score), len(label_score[0]))

    label_predict_ans = []
    for i in range(len(train_dataset_info)):
        label_predict_tmp = np.zeros(28)
        for j in range(28):
            if label_score[i][j] > label_threshold:
                label_predict_tmp[j] = 1
        label_predict_ans += [label_predict_tmp]
    
        #print ('predict = ',type(label_predict_ans), label_predict_ans[i])
        #print ('true = ',label_true[i])
    f1_val = f1_list(label_true,label_predict_ans)
    print ('f1_val = ', f1_val)

def writeList2CSV(res, csvfile):
    with open(csvfile, "w") as output:
        writer = csv.writer(output)
        for val in res:
            writer.writerow(val)   

if __name__ == '__main__':
    app.run(snapshot)


