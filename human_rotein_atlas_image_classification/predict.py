import pandas as pd
from inception_resnet_model import inception_resnet_model

#parameter
INPUT_SHAPE = [299,299]
N_OUT = 28
BATCH_SIZE = 8
submit = pd.read_csv('data/sample_submission.csv')
imagePath = 'data/test/'

model = inception_resnet_model(INPUT_SHAPE,N_OUT,test = False)

model.load('working/model_train.h5')

model.predict(submit,imagePath)
