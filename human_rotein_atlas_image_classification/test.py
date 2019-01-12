import os, sys, math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")

import time
import csv
from uitls import f1,f1_loss,show_history,loss_all,f1_list


def writeList2CSV(res, csvfile):
    with open(csvfile, "w") as output:
        writer = csv.writer(output)
        for val in res:
            writer.writerow(val)   


def readCSV2List(filePath):
    list_ans = []
    with open(filePath, 'rt') as f:
        reader = csv.reader(f)
        for line in reader:
            line = list(line)
            for i in range(len(line)):
                line[i] = float(line[i])
                
            list_ans += [line]
    return list_ans


##label_true = [[0]*28 for i in range(10)]
##print ("label_true",type(label_true),len(label_true), len(label_true[0]))

##writeList2CSV(label_true,'working/test_true.csv')

label_true = readCSV2List('working/label_true.csv')

label_score = readCSV2List('working/label_score.csv')


for k in range(10):
    label_threshold = float(k)/10
    print (label_threshold)
    label_predict_ans = []
    for i in range(len(label_true)):
        label_predict_tmp = np.zeros(28)
        for j in range(28):
            label_true[i][j] = int(label_true[i][j])
            if label_score[i][j] > label_threshold:
                label_predict_tmp[j] = 1
        label_predict_ans += [label_predict_tmp]

        #print ('predict = ',type(label_predict_ans), label_predict_ans[i])
        #print ('true = ',label_true[i])
    print ("label_true",type(label_true),type(label_true[0][0]),len(label_true), len(label_true[0]))
    print ('label_score',type(label_score),type(label_score[0][0]),len(label_score), len(label_score[0]))
    print ('label_predict_ans',type(label_predict_ans),type(label_predict_ans[0][0]),len(label_predict_ans), len(label_predict_ans[0]))
    f1_val = f1_list(label_true,label_predict_ans)
    print ('f1_val = ', f1_val)
