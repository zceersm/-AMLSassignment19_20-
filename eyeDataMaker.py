import numpy as np; import cv2; import glob
sourceFolder = "Datasets/B2"
encoding = {"brown":"0","blue":"1","green":"2","grey":"3","black":"4"}

def makeTrainAndValidateData(cartoonFile,label):
    for f in range(len(cartoonFile)):
        image = cv2.imread(cartoonFile[f])
        cropped = image[240:290, 180:320]
        cropped1D = np.reshape(cropped, np.product(cropped.shape))
        textLine = label +"," + ",".join(map(str, cropped1D.tolist()))
        print(textLine)
        cv2.imshow("cropped", cropped);cv2.waitKey(0)

for col in ["brown","blue","green","grey","black"]:
	listOfFiles = glob.glob(sourceFolder+"/train/"+col+"/*.png") + glob.glob(sourceFolder+"/validate/"+col+"/*.png")
	makeTrainAndValidateData(listOfFiles, encoding[col])
cv2.destroyAllWindows()
