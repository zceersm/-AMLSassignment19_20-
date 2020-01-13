from numpy import loadtxt,argmax,zeros,sum, set_printoptions; from numpy.random import shuffle
from keras.models import Sequential;from keras.layers import Dense, Dropout, LeakyReLU
from keras.utils import to_categorical; import argparse
from keras.layers import Dense, Conv2D, MaxPooling2D, BatchNormalization, GlobalAveragePooling2D
from keras import models; import numpy as np; import cv2; import glob
from config import *
#epoch 14
#cropped picture dimension 140x50
def makeDataB2(folder):
    eyeAr = {}
    sourceFolder = dataFolder + "B2" + dl
    encoding = {"brown":0,"blue":1,"green":2,"grey":3,"black":4}
    for col in ["brown","blue","green","grey","black"]:
        Folder = sourceFolder+folder+dl+col+dl+"*.png"
        File = (glob.glob(sourceFolder+folder+dl+col+dl+"*.png"))
        eyeAr[col] = np.zeros([len(File),3*140*50+1])
        for f in range(len(File)):
            R = np.array([],dtype=np.uint8)
            image = cv2.imread(File[f])
            #crop images leaving just the eyes area
            #feed croppeds into the neural network
            #cropped picture dimension 140x50
            cropped = image[240:290, 180:320]
            cropped1D = np.reshape(cropped, np.product(cropped.shape))
            R = np.append(R,cropped1D)
        addL = np.insert(R, 0, encoding[col])
        eyeAr[col][f,:] = addL[:]
    return(np.vstack([eyeAr["brown"],eyeAr["blue"],eyeAr["green"],eyeAr["grey"],eyeAr["black"]]))

def data_preprocessingB2():
    getData = {}
    getData["train"] = makeDataB2("train")
    getData["validate"] = makeDataB2("validate")
    getData["test"] = makeDataB2("test")
    return(getData["train"],getData["validate"],getData["test"])

def buildB2():
    numberOfInputs = 3*140*50
    ram_model = Sequential()
    ram_model.add(Dense(200, input_dim=numberOfInputs, activation='relu'))
    ram_model.add(Dense(150, activation='relu'))
    ram_model.add(Dense(100, activation='relu'))
    ram_model.add(Dense(50, activation='relu'))
    ram_model.add(Dense(5, activation='softmax'))
    ram_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return(ram_model)

def trainB2(model_B2, data_trainB2, data_valB2):
    X = data_trainB2[:,1:]; Y = to_categorical(data_trainB2[:,0])
    Xv =  data_valB2[:,1:]; Yv = to_categorical(data_valB2[:,0])
    history = model_B2.fit(X, Y, validation_data = (Xv,Yv), epochs=14, batch_size=10)
    return(history.history['accuracy'][13])
    
def testB2(model_B2, data_testB2):
    Xt = data_testB2[:,1:]; Yt = to_categorical(data_testB2[:,0])
    metrics = model_B2.evaluate(Xt,Yt,batch_size = 10)
    return(metrics[1])

