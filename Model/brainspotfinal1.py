#import and lode modules
import numpy as np
import matplotlib.pyplot as plt
import os
import math
import shutil
import glob

#from google.colab import drive
#drive.mount('/content/drive')

Root="/content/drive/MyDrive/NewDataset3/Brain Tumour1"
number_of_images={}
for dir in os.listdir(Root):
  number_of_images[dir]=len(os.listdir(os.path.join(Root,dir)))

number_of_images.items()

#Split the data -> 70% for train, 15% for validation, 15% for testing

def dataFolder(p,split):

  if not os.path.exists("./"+p):
    os.mkdir("./"+p)

    for dir in os.listdir(Root):
      os.makedirs("./"+p+"/"+dir)

      for img in np.random.choice(a=os.listdir(os.path.join(Root,dir)),
                                  size=(math.floor(split*number_of_images[dir])-5),
                                  replace=False):

        O=os.path.join(Root,dir,img)
        D=os.path.join("./"+p,dir)
        shutil.copy(O,D)
        #os.remove(O)

  else:
    print( f"{p}folder exists")

dataFolder("train",0.7)

dataFolder("validation",0.15)

dataFolder("test",0.15)

#Model build

from keras.layers import Conv2D, MaxPool2D,Dropout,Flatten,Dense,BatchNormalization,GlobalAvgPool2D
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
import keras

import keras
print(keras.__version__)