from numpy import loadtxt,zeros,argmax,sum;from numpy.random import shuffle
from keras.utils import to_categorical; from keras.models import load_model
import imutils, numpy as np, time, dlib, cv2
from scipy.spatial import distance as dist
from imutils import face_utils;import glob

savedModel = "model.h5";ram_Model = load_model(savedModel)
shapePredictor= "shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(shapePredictor)
(mStart, mEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]

def testFiles(fileList,realAnswer):
	rightAnswer = 0; wrongAnswer = 0
	for file in fileList:
		image = cv2.imread(file)
		gimage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		rectangleMark = detector(gimage, 0)
		if	len(rectangleMark) == 1:
			shape = predictor(gimage, rectangleMark[0])
			shape = face_utils.shape_to_np(shape)
			lips = shape[mStart:mEnd]
			l = min(lips[:,0]);r=max(lips[:,0])
			t = min(lips[:,1]);b=max(lips[:,1])
			w = r-l
			h = b-t
			biggest = max(w,h)		
			lips[:,0] = lips[:,0]-l
			lips[:,1] = lips[:,1]-t
			for point in range(lips.shape[0]):
				for co in range(lips.shape[1]):
					lips[point,co] = int(lips[point,co] * 100/biggest)
			xarr = lips.reshape([40])
			xarr = xarr.reshape([1,40])
			if ram_Model.predict_classes(xarr)[0] == realAnswer:
				rightAnswer = rightAnswer+1
			else:
				wrongAnswer = wrongAnswer+1
	return rightAnswer, wrongAnswer

rootFolder = 'Datasets/A2/'
smile=(glob.glob(rootFolder + "test/yes/*.jpg"))
nonSmile=(glob.glob(rootFolder + "test/no/*.jpg"))
tPos,fNeg = testFiles(smile,0)
tNeg, fPos = testFiles(nonSmile,1)
total = tPos + tNeg + fPos + fNeg
print(tPos,tNeg,fPos, fNeg)
print("Accuracy: %0.1f%%"%((tPos+tNeg)*100/total))




"""
ram_model = "model.h5"; myModel = load_model(ram_model)
shapePredictor= "shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(shapePredictor)
(mStart, mEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]

def testFiles(fileList,correctAnswer):
    right = 0; wrong = 0
    for file in fileList:
        image = cv2.imread(file)
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
            xarr = lips.reshape([40])
            xarr = xarr.reshape([1,40])
            if myModel.predict_classes(xarr)[0] == correctAnswer:
                right = right+1
            else:
                wrong = wrong+1
    return right, wrong

smile=(glob.glob("/Users/faymita29/Desktop/AMLS/AMLS_19-20_SN16015140/Datasets/A2/test/yes/*.jpg"))
nonSmile=(glob.glob("/Users/faymita29/Desktop/AMLS/AMLS_19-20_SN16015140/Datasets/A2/test/no/*.jpg"))

truePos,falseNeg = testFiles(smile,0)
trueNeg, falsePos = testFiles(nonSmile,1)
total = truePos + trueNeg + falsePos + falseNeg
print(truePos,trueNeg,falsePos, falseNeg)
print("Accuracy: %0.1f%%"%((truePos+trueNeg)*100/total))

"""