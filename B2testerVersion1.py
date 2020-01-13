import numpy as np; import cv2; import glob
from keras.models import load_model

savedModel = "model.h5"
Model = load_model(savedModel)
rootFolder = "/Users/faymita29/Desktop/AMLS/AMLS_19-20_SN16015140/Datasets/B2"
encoding = {"brown":"0","blue":"1","green":"2","grey":"3","black":"4"}

def makeTest(cartoonFile,realAnswer):
    rightAnswer = 0; wrongAnswer = 0
    for f in range(len(cartoonFile)):
        image = cv2.imread(cartoonFile[f])
        cropped = image[240:290, 180:320]
        cropped1D = np.reshape(cropped, np.product(cropped.shape))
        #textLine = realAnswer +"," + ",".join(map(str, cropped1D.tolist()))
        #print(textLine)
        

for col in ["brown","blue","green","grey","black"]:
	listOfFiles = glob.glob(rootFolder+"/test/"+col+"/0.png")
	makeTest(listOfFiles, encoding[col])

"""
import math, cv2, glob,sys; import numpy as np
#from keras.models import load_model

#modelFile = "eyeModel.h5"
#myModel = load_model(modelFile)

testfile = open("eyeDataForTest.csv","r")
testLines = testfile.readlines()
numRows = len(testLines)
numCols = len(testLines[0])
testData = np.array([numRows,numCols])
for l in range(numRows):
    testData[l] = np.toarray(testLines[l].split(","))

"""