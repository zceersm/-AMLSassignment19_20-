from numpy import loadtxt,argmax,zeros,sum, set_printoptions; from numpy.random import shuffle
from keras.models import Sequential;from keras.layers import Dense, Dropout, LeakyReLU
from keras.utils import to_categorical; import argparse
from keras.layers import Dense, Conv2D, MaxPooling2D, BatchNormalization, GlobalAveragePooling2D
import imutils, numpy as np, time, cv2, dlib
from scipy.spatial import distance as dist
from imutils import face_utils
import glob
from keras import models
from config import *
#epoch 50

def makeDataA2(folder):
    shape_predictor= "A2"+dl+"shape_predictor_68_face_landmarks.dat"
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(shape_predictor)
    (mStart, mEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]
    sourceFolder = dataFolder + "A2" + dl
    encodes = {"yes":0,"no":1}
    emotionarrays = {}
    for emotion in ["yes","no"]:
        files = glob.glob(sourceFolder+folder+dl+emotion+dl+"*.jpg")
        emotionarrays[emotion] = np.zeros([len(files),41])
        for i in range(len(files)):
            image = cv2.imread(files[i])
            gimage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            rectangleMark = detector(gimage, 0)
            if len(rectangleMark)==1:
                shape = predictor(gimage, rectangleMark[0])
                shape = face_utils.shape_to_np(shape)
                lips = shape[mStart:mEnd]
                left = min(lips[:,0])
                right = max(lips[:,0])
                top = min(lips[:,1])
                bottom = max(lips[:,1])
                w = right - left
                h = bottom - top
                b = max(w, h)
                lips[:,0] = lips[:,0]-left 
                lips[:,1] = lips[:,1]-top
                for point in range(lips.shape[0]):
                    for co in range(lips.shape[1]):
                        lips[point,co]=int(lips[point,co] * 100/b)
                        
                lips1D = lips.reshape(1,40)
                withlabel = np.insert(lips1D,0,encodes[emotion])
                emotionarrays[emotion][i,:] = withlabel[:]
    return(np.vstack([emotionarrays["yes"],emotionarrays["no"]]))

def data_preprocessingA2():
    getData = {}
    getData["train"] = makeDataA2("train")
    getData["validate"] = makeDataA2("validate")
    getData["test"] = makeDataA2("test")
    return(getData["train"],getData["validate"],getData["test"])

def buildA2():
    numberOfInputs = 40
    ram_model = Sequential()
    ram_model.add(Dense(100, input_dim=numberOfInputs, activation='relu'))
    ram_model.add(Dense(80, activation='relu'))
    ram_model.add(Dense(50, activation='relu'))
    ram_model.add(Dense(20, activation='relu'))
    ram_model.add(Dropout(0.5))
    ram_model.add(Dense(1, activation='sigmoid'))
    ram_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return(ram_model)

def trainA2(model_A2, data_trainA2, data_valA2):
    X = data_trainA2[:,1:]
    Y = data_trainA2[:,0]
    Xv =  data_valA2[:,1:]
    Yv = data_valA2[:,0]
    history = model_A2.fit(X, Y, validation_data = (Xv,Yv), epochs=50, batch_size=20)
    return(history.history['accuracy'][49])

def testA2(model_A2, data_testA2):
    batchSize= 16
    Xt = data_testA2[:,1:]
    Yt = data_testA2[:,0]
    metrics = model_A2.evaluate(Xt,Yt,batch_size = 20)
    return(metrics[1])
    