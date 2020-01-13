import numpy as np; import cv2; import glob
from keras.models import load_model
from keras.preprocessing.image import img_to_array, load_img, ImageDataGenerator

savedModel = "model.h5"
B1Model = load_model(savedModel)
rootFolder = "AMLS/AMLS_19-20_SN16015140/Datasets/B1"

def makeTest(cartoonFile,realAnswer):
    rightAnswer = 0; wrongAnswer = 0
    for s in cartoonFile:
        testImg = load_img(s)
        testArr = img_to_array(testImg)
        testArr = np.expand_dims(testArr, axis =0)
        prediction = B1Model.predict_classes(testArr)
        if prediction == realAnswer :
            rightAnswer +=1
        else:
            wrongAnswer +=1
    return rightAnswer, wrongAnswer

overallRights = 0; overallWrongs = 0	

encoding = {"0":0,"1":1,"2":2,"3":3,"4":4}
for col in ["0","1","2","3","4"]:
	listOfFiles = glob.glob(rootFolder+"/test/"+col+"/*.png")
	rightAnswer, wrongAnswer = makeTest(listOfFiles, encoding[col])
	totalAnswers = rightAnswer+wrongAnswer
	Accuracy = 100*rightAnswer/totalAnswers
	overallRights +=rightAnswer; overallWrongs +=wrongAnswer
	print("%s: %d right, %d wrong (Accuracy %0.1f%%)"%(col,rightAnswer,wrongAnswer,Accuracy))
    
overallTotal = overallRights + overallWrongs
overallAccuracy = 100*overallRights/overallTotal
print("Overall: %d right, %d wrong (Accuracy %0.1f%%)"%(overallRights,overallWrongs,overallAccuracy))