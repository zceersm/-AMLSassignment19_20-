from keras.layers import Dense, Conv2D, MaxPooling2D, BatchNormalization, GlobalAveragePooling2D
from keras import models
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img, ImageDataGenerator
from keras.callbacks import EarlyStopping
from config import *
from keras.models import load_model
import glob, sys; import numpy as np; import cv2

#epoch 50
def buildA1():
    # starting point 
    ram_model= models.Sequential()
    # 1st convolutional block
    ram_model.add(Conv2D(16, (3, 3), activation='relu', padding='same', input_shape=(218,178,3)))
    ram_model.add(MaxPooling2D((2, 2), padding='same'))
    # 2nd block
    ram_model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
    ram_model.add(MaxPooling2D((2, 2), padding='same'))
    # 3rd block
    ram_model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    ram_model.add(MaxPooling2D((2, 2), padding='same'))
    # 4th block
    ram_model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    ram_model.add(MaxPooling2D((2, 2), padding='same'))
    # global average pooling
    ram_model.add(GlobalAveragePooling2D())
    # completely connected layer
    ram_model.add(Dense(64, activation='relu'))
    ram_model.add(BatchNormalization())
    #predicting
    ram_model.add(Dense(1, activation='sigmoid'))
    ram_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return(ram_model)

def makeDataA1():
    #setting data generator up
    rootFolder = dataFolder + "A1" + dl
    Image_data_generator = ImageDataGenerator()
    # get training images from my directory in batches
    train_image_generator = Image_data_generator.flow_from_directory(rootFolder+"train", target_size=(218, 178), batch_size=10, class_mode='binary', shuffle=True)
    # get validation images from my directory in batches
    validation_image_generator = Image_data_generator.flow_from_directory(rootFolder+"validate", target_size=(218, 178), batch_size=10, class_mode='binary', shuffle=True)
    return(train_image_generator, validation_image_generator)

def trainA1(model_A1):
    train_image_generator, validation_image_generator = makeDataA1()
    history = model_A1.fit_generator(train_image_generator, epochs=50, steps_per_epoch=100, validation_data=validation_image_generator)
    return(history.history['accuracy'][49])

def testA1(model_A1):
    root = dataFolder + "A1" + dl + "test" + dl
    female=glob.glob(root + "female" + dl+ "*.jpg")
    male=glob.glob(root + "male" + dl + "*.jpg")
    fright = 0; fwrong = 0;
    mright = 0; mwrong = 0;
    for f in female:
        testImg = load_img(f)
        testArr = img_to_array(testImg)
        testArr = np.expand_dims(testArr,axis=0)
        prediction = model_A1.predict_classes(testArr)[0][0]
        if prediction == 0:
            fright +=1
        else:
            fwrong +=1
    for m in male:
        testImg = load_img(m)
        testArr = img_to_array(testImg)
        testArr = np.expand_dims(testArr,axis=0)
        prediction = model_A1.predict_classes(testArr)[0][0]
        if prediction == 1: 
            mright +=1
        else:
            mwrong +=1
    overallAccuracy = (mright+fright)/(mright+mwrong+fright+fwrong)
    return(overallAccuracy)