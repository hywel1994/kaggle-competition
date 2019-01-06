import pandas as pd
from inception_resnet_model import inception_resnet_model

#parameter
INPUT_SHAPE = [299,299]
N_OUT = 28

submit = pd.read_csv('data/sample_submission.csv')
imagePath = 'data/test/'

model = inception_resnet_model(INPUT_SHAPE,N_OUT,div=3)

model.load('working/model_train1546691056.h5')
model.load_weight('working/log_1_5/test-22.hdf5')
model.predict(submit,imagePath,type_image=0)
