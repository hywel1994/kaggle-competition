from tensorflow import keras
import tensorflow as tf
from keras.metrics import categorical_accuracy
from keras.callbacks import LearningRateScheduler
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

from keras.callbacks import TensorBoard
from keras.callbacks import LearningRateScheduler
from keras import regularizers

import os, sys, math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

#from imgaug import augmenters as iaa
from tqdm import tqdm

from uitls import f1,f1_loss,show_history,focal_loss,loss_all
from kaggle_data import data_generator  
from sgdr_callback import SGDRScheduler


class XTensorBoard(TensorBoard):
    def on_epoch_begin(self, epoch, logs=None):
        # get values
        lr = float(K.get_value(self.model.optimizer.lr))
        decay = float(K.get_value(self.model.optimizer.decay))
        # computer lr
        lr = lr * (1. / (1 + decay * epoch))
        K.set_value(self.model.optimizer.lr, lr)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        logs['lr'] = K.get_value(self.model.optimizer.lr)
        super().on_epoch_end(epoch, logs)


class inception_resnet_model:
    def __init__(self, input_shape, n_out, div=3, scheduler_type=None, test=False, log_dir ='./log'):
        self.test = test
        self.input_shape = input_shape
        self.n_out = n_out
        self.div = div
        self.scheduler_type=scheduler_type
        self.log_dir = log_dir

    def create_model(self):
        self.pretrain_model = InceptionResNetV2(include_top=False, weights='imagenet', input_shape=self.input_shape+[3])
        if self.div==3:
            input_tensor = Input(shape=self.input_shape+[3])
            bn = BatchNormalization()(input_tensor)
            #out = Conv2D(3, kernel_size=1, strides=1, padding='same', activation='tanh', kernel_regularizer=regularizers.l2(1e-4))(input_tensor)
        elif self.div==4:
            input_tensor = Input(shape=self.input_shape+[4])
            bn = Conv2D(3, kernel_size=1, strides=1, padding='same', activation='tanh', kernel_regularizer=regularizers.l2(1e-4))(input_tensor)
            bn = BatchNormalization()(bn)
        else:
            print('image div error')
            return
        
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
        def lr_schedule(epoch):
            if epoch % 10 == 0 and epoch != 0:
                lr = K.get_value(self.model.optimizer.lr)
                K.set_value(self.model.optimizer.lr, lr * 0.1)
                print("lr changed to {}".format(lr * 0.1))
            return K.get_value(self.model.optimizer.lr)

        if self.test:
            epoch = 2
        else:
            if trainable:
                epoch = 40
            else:
                epoch = 20
        print ('self.log_dir = ', self.log_dir)
        if self.scheduler_type=='sgdr':
            scheduler = SGDRScheduler(min_lr=1e-6,
                                max_lr=1e-3,
                                steps_per_epoch=1000,
                                lr_decay=0.9,
                                cycle_length=5,
                                mult_factor=1.5)
            callback_list=[scheduler, XTensorBoard(log_dir=self.log_dir)]
        elif self.scheduler_type=='lr':
            scheduler = LearningRateScheduler(lr_schedule)
            callback_list=[scheduler, XTensorBoard(log_dir=self.log_dir)]
        else:
            callback_list=[XTensorBoard(log_dir=self.log_dir)]

        #self.inception_resnet_trainable(trainable)
        history = self.model.fit_generator(
            generator=self.training_generator,
            steps_per_epoch=1000,
            validation_data=next(self.validation_generator),
            epochs=epoch, 
            verbose=1,
            callbacks=callback_list)
        
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


