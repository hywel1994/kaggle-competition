from tensorflow import keras
import tensorflow as tf
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

from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras import regularizers

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
    def __init__(self, log_dir):  # add other arguments to __init__ if you need
        super().__init__(log_dir=log_dir)

    def on_batch_end(self, batch, logs=None):
        logs.update({'lr_batch': K.eval(self.model.optimizer.lr)})
        super().on_batch_end(batch, logs)

    def on_epoch_end(self, epoch, logs=None):
        logs.update({'lr_epoch': K.eval(self.model.optimizer.lr)})
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
        #self.pretrain_model = InceptionResNetV2(include_top=False, weights='imagenet', input_shape=self.input_shape+[3])
        self.pretrain_model = InceptionResNetV2(include_top=False, weights='imagenet', input_shape=self.input_shape+[3],pooling='avg',classes='avg')

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
        '''
        x = Conv2D(128, kernel_size=(1,1), activation='relu')(x)
        x = Flatten()(x)
        x = Dropout(0.5)(x)
        x = Dense(512, activation='relu')(x)
        x = Dropout(0.5)(x)
        '''
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
            steps_per_epoch = 10
        else:
            steps_per_epoch = 3000
            if trainable:
                epoch = 200
            else:
                epoch = 5
        filepath="test-{epoch:02d}.hdf5"
        checkpoint = ModelCheckpoint(filepath, monitor='f1', verbose=1, save_best_only=True, mode='max')
        #print ('self.log_dir = ', self.log_dir)
        if self.scheduler_type=='sgdr':
            scheduler = SGDRScheduler(min_lr=1e-6/4,
                                max_lr=1e-3/4,
                                steps_per_epoch=steps_per_epoch,
                                lr_decay=0.9,
                                cycle_length=10,
                                mult_factor=1.5)
            callback_list=[scheduler,XTensorBoard(log_dir=self.log_dir),checkpoint]
        elif self.scheduler_type=='lr':
            scheduler = LearningRateScheduler(lr_schedule)
            callback_list=[scheduler, XTensorBoard(log_dir=self.log_dir),checkpoint]
        else:
            callback_list=[XTensorBoard(log_dir=self.log_dir),checkpoint]

        #self.inception_resnet_trainable(trainable)
        history = self.model.fit_generator(
            generator=self.training_generator,
            steps_per_epoch=steps_per_epoch,
            validation_data=next(self.validation_generator),
            epochs=epoch, 
            verbose=1,
            callbacks=callback_list)
        
        return history
    
    def predict(self, submit,imagePath, type_image):
        predicted = []
        for name in tqdm(submit['Id']):
            path = os.path.join(imagePath, name)
            image = data_generator.load_image(path, self.input_shape+[self.div],type_image)
            score_predict = self.model.predict(image[np.newaxis])[0]
            label_predict = np.arange(28)[score_predict>=0.2]
            str_predict_label = ' '.join(str(l) for l in label_predict)
            predicted.append(str_predict_label)

        submit['Predicted'] = predicted
        submit.to_csv('submission.csv', index=False)
    
    def eval(self, dataset_info, type_image):
        labels_true = []
        scores_predict = []
        for idx in range(len(dataset_info)):
            image = data_generator.load_image(dataset_info[idx]['path'], self.input_shape+[self.div], type_image)   
            labels = np.zeros(28)
            labels[dataset_info[idx]['labels']] = 1
            labels_true +=[labels]
            score_predict = self.model.predict(image[np.newaxis])[0]
            #label_predict = np.arange(28)[score_predict>=label_threshold]
            scores_predict += [score_predict]

        return labels_true, scores_predict
    
    def snapshot_true(self, dataset_info):
        labels_true = []
        for idx in range(len(dataset_info)):
            labels = np.zeros(28)
            labels[dataset_info[idx]['labels']] = 1
            labels_true +=[labels]

        return labels_true
    
    def snapshot_val(self, dataset_info, type_image):
        scores_predict = []

        for idx in range(len(dataset_info)):
            image = data_generator.load_image(dataset_info[idx]['path'], self.input_shape+[self.div], type_image)   
            score_predict = self.model.predict(image[np.newaxis])[0]
            scores_predict += [score_predict]
        return scores_predict
    
    def snapshot_pre(self, submit,imagePath, type_image):
        scores_predict = []
        for name in tqdm(submit['Id']):
            path = os.path.join(imagePath, name)
            image = data_generator.load_image(path, self.input_shape+[self.div],type_image)
            score_predict = self.model.predict(image[np.newaxis])[0]
            scores_predict += [score_predict]
        return score_predict


    def save(self, modeloutputpath):
        self.model.save(modeloutputpath)
    
    def load(self, modelinputpath):
        self.model = load_model(modelinputpath,custom_objects={'loss_all':loss_all, 'focal_loss': focal_loss, 'f1':f1})

    def load_weight(self, modelinputpath):
        self.model.load_weights(modelinputpath)

    def inception_resnet_trainable(self,trainable):
        for layer in self.pretrain_model.layers:
            layer.trainable = trainable


