import numpy as np; import cv2; import glob
from keras.models import load_model
from sklearn.metrics import confusion_matrix

savedModel = "model.h5"
B1Model = load_model(savedModel)
rootFolder = "Datasets/B2"
encoding = {"brown":0,"blue":1,"green":2,"grey":3,"black":4}
t = []
p = []

def makeTest(cartoonFile,realAnswer):
    rightAnswer = 0; wrongAnswer = 0
    for f in range(len(cartoonFile)):
        image = cv2.imread(cartoonFile[f])
        cropped = image[240:290, 180:320]
        cropped1D = np.reshape(cropped, np.product(cropped.shape))
        cropped1D = np.expand_dims(cropped1D, axis =0)
        prediction = B1Model.predict_classes(cropped1D)
        t.append(realAnswer)
        p.append(prediction)
        if prediction == realAnswer:
            rightAnswer +=1
        else:
            wrongAnswer +=1
    return rightAnswer, wrongAnswer

overallRights = 0; overallWrongs = 0	
for col in ["brown","blue","green","grey","black"]:
	listOfFiles = glob.glob(rootFolder+"/test/"+col+"/*.png")
	rightAnswer, wrongAnswer = makeTest(listOfFiles, encoding[col])
	totalAnswers = rightAnswer+wrongAnswer
	Accuracy = 100*rightAnswer/totalAnswers
	overallRights +=rightAnswer; overallWrongs +=wrongAnswer
	print("%s: %d right, %d wrong (Accuracy %0.1f%%)"%(col,rightAnswer,wrongAnswer,Accuracy))
overallTotal = overallRights + overallWrongs
overallAccuracy = 100*overallRights/overallTotal
print("Overall: %d right, %d wrong (Accuracy %0.1f%%)"%(overallRights,overallWrongs,overallAccuracy))

cm = confusion_matrix(t,p)
print(cm)