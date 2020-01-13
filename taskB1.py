from keras.layers import Dense, Conv2D, MaxPooling2D, BatchNormalization, GlobalAveragePooling2D
from keras import models
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from keras.callbacks import EarlyStopping
from config import *
import numpy as np; import cv2; import glob
from keras.models import load_model

#epoch 10
def buildB1():
    # starting point 
    ram_model= models.Sequential()
    # 1st convolutional block
    ram_model.add(Conv2D(24, (3, 3), activation='relu', padding='same', input_shape=(500,500,3)))
    ram_model.add(MaxPooling2D((2, 2), padding='same'))
    # 2nd block
    ram_model.add(Conv2D(24, (3, 3), activation='relu', padding='same'))
    ram_model.add(MaxPooling2D((2, 2), padding='same'))
    # 3rd block
    ram_model.add(Conv2D(48, (3, 3), activation='relu', padding='same'))
    ram_model.add(MaxPooling2D((2, 2), padding='same'))
    # 4th block
    ram_model.add(Conv2D(96, (3, 3), activation='relu', padding='same'))
    ram_model.add(MaxPooling2D((2, 2), padding='same'))
    # global average pooling
    ram_model.add(GlobalAveragePooling2D())
    #predicting
    ram_model.add(Dense(5, activation='softmax'))
    # compile model 
    ram_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return(ram_model)

def makeDataB1():
    #setting data generator up
    rootFolder = dataFolder + "B1" + dl
    Image_data_generator = ImageDataGenerator()
    # get training images from my directory in batches
    train_image_generator = Image_data_generator.flow_from_directory(rootFolder+"train",target_size=(500, 500),batch_size=10, class_mode='categorical', shuffle=True)
    # get validation images from my directory in batches
    validation_image_generator = Image_data_generator.flow_from_directory(rootFolder+"validate",target_size=(500, 500), batch_size=10, class_mode='categorical', shuffle=True)
    return(train_image_generator, validation_image_generator)

def trainB1(model_B1):
    train_image_generator, validation_image_generator = makeDataB1()
    history = model_B1.fit_generator(train_image_generator,epochs=10,steps_per_epoch=100, validation_data=validation_image_generator)
    return(history.history['accuracy'][9])

def testB1(model_B1):
    rootFolder = dataFolder + "B1" + dl
    overallRights = 0; overallWrongs = 0
    encoding = {"0":0,"1":1,"2":2,"3":3,"4":4}
    for col in ["0","1","2","3","4"]:
        cartoonFile = glob.glob(rootFolder+"test"+dl+col+dl+"*.png")
        rightAnswer = 0; wrongAnswer = 0
        for s in cartoonFile:
            testImg = load_img(s)
            testArr = img_to_array(testImg)
            testArr = np.expand_dims(testArr, axis =0)
            prediction = model_B1.predict_classes(testArr)
            if prediction == encoding[col] :
                rightAnswer +=1
            else:
                wrongAnswer +=1
        overallRights += rightAnswer
        overallWrongs += wrongAnswer
    overallTotal = overallRights + overallWrongs
    overallAccuracy = overallRights/overallTotal
    return(overallAccuracy)