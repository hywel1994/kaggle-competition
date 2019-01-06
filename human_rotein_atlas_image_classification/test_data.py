import numpy as np
from PIL import Image


path = 'data/train/3ea01024-bbc9-11e8-b2bc-ac1f6b6435d0'
#R = np.array(Image.open(path+'_red.png')).reshape(512,512)
G = np.array(Image.open(path+'_green.png')).reshape(512,512)

im = Image.open(path+'_red.png')
im.show()
