#******Testing *********************

from keras.preprocessing.image import img_to_array, load_img, ImageDataGenerator
from keras.models import load_model
import glob, sys; import numpy as np; import cv2

savedModel = load_model("model.h5")
root='Datasets/A1/test/'
female=(glob.glob(root + "female/*.jpg"))
male=(glob.glob(root+"male/*.jpg"))
fright = 0; fwrong = 0; fpredictions = []
mright = 0; mwrong = 0; mpredictions = []

for f in female:
	testImg = load_img(f)
	testArr = img_to_array(testImg)
	testArr = testArr.reshape((1,)+testArr.shape)
	prediction = savedModel.predict_classes(testArr)[0][0]
	fpredictions.append(prediction)
	if prediction == 0: 
		fright +=1
	else:
		fwrong +=1

for m in male:
	testImg = load_img(f)
	testArr = img_to_array(testImg)
	testArr = testArr.reshape((1,)+testArr.shape)
	prediction = savedModel.predict_classes(testArr)[0][0]
	mpredictions.append(prediction)
	if prediction == 0: 
		mright +=1
	else:
		mwrong +=1
        
print(fright, fwrong,mright,mwrong);print("Overall accuracy: %0.2f%%"%(100*(mright+fright)/(mright+mwrong+fright+fwrong)))
print("Women: %0.2f%%"%(100*fright/(fright+fwrong)));print("Men: %0.2f%%"%(100*mright/(mright+mwrong)))
