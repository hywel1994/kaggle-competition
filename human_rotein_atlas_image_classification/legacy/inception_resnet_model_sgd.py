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
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.callbacks import LearningRateScheduler

import os, sys, math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

#from imgaug import augmenters as iaa
from tqdm import tqdm

from uitls import f1,f1_loss,show_history,focal_loss,loss_all
from kaggle_data_3 import data_generator
from sgdr_callback import SGDRScheduler


class inception_resnet_model:
    def __init__(self, input_shape, n_out, test=False):
        self.test = test
        self.input_shape = input_shape
        self.n_out = n_out

    def create_model(self):
        self.pretrain_model = InceptionResNetV2(include_top=False, weights='imagenet', input_shape=self.input_shape+[3])

        input_tensor = Input(shape=self.input_shape+[3])
        #out = Conv2D(3, kernel_size=1, strides=1, padding='same', activation='tanh', kernel_regularizer=regularizers.l2(1e-4))(input_tensor)
        bn = BatchNormalization()(input_tensor)
        x = self.pretrain_model(bn)
        
        x = Conv2D(128, kernel_size=(1,1), activation='relu')(x)
        x = Flatten()(x)
        x = Dropout(0.5)(x)
        x = Dense(512, activation='relu')(x)
        x = Dropout(0.5)(x)

        output = Dense(self.n_out, activation='sigmoid')(x)
        
        self.model = Model(input_tensor, output)

    def compile_model(self):
        self.model.compile(
            loss=loss_all,  
            optimizer=Adam(1e-3),
            metrics=['categorical_accuracy', 'binary_accuracy', focal_loss, f1])

    def summary(self):
        self.model.summary()
    
    def set_generators(self, train_generator, validation_generator):
        self.training_generator = train_generator
        self.validation_generator = validation_generator

    def learn(self,trainable = True):
        if self.test:
            epoch = 2
        else:
            if trainable:
                epoch = 40
            else:
                epoch = 20
                
        schedule = SGDRScheduler(min_lr=1e-5,
                                max_lr=1e-2,
                                steps_per_epoch=1000,
                                lr_decay=0.9,
                                cycle_length=5,
                                mult_factor=1.5)

        #self.inception_resnet_trainable(trainable)
        history = self.model.fit_generator(
            generator=self.training_generator,
            steps_per_epoch=1000,
            validation_data=next(self.validation_generator),
            epochs=epoch, 
            verbose=1,
            callbacks=[schedule,TensorBoard(log_dir='./log')])
        
        return history
    
    def predict(self, submit,imagePath):
        predicted = []
        for name in tqdm(submit['Id']):
            path = os.path.join(imagePath, name)
            image = data_generator.load_image(path, self.input_shape)
            score_predict = self.model.predict(image[np.newaxis])[0]
            label_predict = np.arange(28)[score_predict>=0.2]
            str_predict_label = ' '.join(str(l) for l in label_predict)
            predicted.append(str_predict_label)

        submit['Predicted'] = predicted
        submit.to_csv('submission.csv', index=False)

    def save(self, modeloutputpath):
        self.model.save(modeloutputpath)
    
    def load(self, modelinputpath):
        self.model = load_model(modelinputpath,custom_objects={'loss_all':loss_all, 'focal_loss': focal_loss, 'f1':f1})

    def inception_resnet_trainable(self,trainable):
        for layer in self.pretrain_model.layers:
            layer.trainable = trainable


