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

#CNN model

from keras.models import Sequential  # Make sure to import Sequential with an uppercase "S"
from keras.layers import Conv2D, MaxPool2D, Dropout, Flatten, Dense

model = Sequential()

model.add(Conv2D(filters=16, kernel_size=(3, 3), activation='relu', input_shape=(224, 224, 3), padding='same'))
model.add(Conv2D(filters=36, kernel_size=(3, 3), activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))

model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))

model.add(Conv2D(filters=128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))

model.add(GlobalAvgPool2D())  # Add Global Average Pooling layer

model.add(Dropout(rate=0.25))

model.add(Dense(units=128, activation='relu'))
model.add(Dropout(rate=0.25))
model.add(Dense(units=1, activation='sigmoid'))

model.summary()



model.compile(optimizer='adam',loss=keras.losses.binary_crossentropy, metrics=['accuracy'])

#use data generator

def preImage1(path):
  """
  input:path
  output:pre proccessed images
  """
  image_data=ImageDataGenerator(zoom_range=0.2,shear_range=0.2,rescale=1/255,horizontal_flip=True)  #data augmentation

  image=image_data.flow_from_directory(directory=path,target_size=(244,244),batch_size=32,class_mode='binary')

  return image

from tensorflow.python.util.compat import path_to_str
path="/content/train"

train_data=preImage1(path)

#use data generator

def preImage2(path):
  """
  input:path
  output:pre proccessed images
  """
  image_data=ImageDataGenerator(rescale=1/255)#no data augmentation in testing

  image=image_data.flow_from_directory(directory=path,target_size=(244,244),batch_size=32,class_mode='binary')

  return image

path="/content/test"

test_data=preImage2(path)

path="/content/validation"

val_data=preImage2(path)

#early stopping and model check point

from keras.callbacks import ModelCheckpoint,EarlyStopping

#early stopping

es=EarlyStopping(monitor="val_accuracy",min_delta=0.01,patience=12,verbose=1,mode='auto')