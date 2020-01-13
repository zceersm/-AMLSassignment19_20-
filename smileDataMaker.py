import imutils, numpy as np, time, dlib, cv2
from scipy.spatial import distance as dist
from imutils import face_utils;import glob

rootFolder = "/Users/faymita29/Desktop/AMLS/AMLS_19-20_SN16015140/Datasets/A2/"
smile = (glob.glob(rootFolder + "train/yes/*.jpg"))+(glob.glob(rootFolder + "validate/yes/*.jpg"))
nonSmile = (glob.glob(rootFolder + "train/no/*.jpg"))+(glob.glob(rootFolder + "validate/no/*.jpg"))

cv2.namedWindow('img',cv2.WINDOW_AUTOSIZE)
shapePredictor= "shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(shapePredictor)
(mStart, mEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]

for file in smile:
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
        cv2.rectangle(image,(left,top),(right,bottom),(255,255,255),2)
        for i in range(lips.shape[0]-1):
            cv2.line(image,(lips[i][0],lips[i][1]),(lips[i+1][0],lips[i+1][1]),(255,255,255),1)
        print("0,"+",".join(map(str,lips.reshape([40]))))
        cv2.imshow('img',image)
        cv2.waitKey(1)

for file in nonSmile:
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
        cv2.rectangle(image,(left,top),(right,bottom),(255,255,255),2)
        for i in range(lips.shape[0]-1):
            cv2.line(image,(lips[i][0],lips[i][1]),(lips[i+1][0],lips[i+1][1]),(255,255,255),1)
        print("1,"+",".join(map(str,lips.reshape([40]))))
        cv2.imshow('img',image)
        cv2.waitKey(1)
        
cv2.destroyAllWindows()
        