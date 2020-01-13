import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import random
import pickle

DATADIR = "D:\\cat"
CATEGORIES = ["grumpy_cat", "stubbs"]

for category in CATEGORIES:
    path = os.path.join(DATADIR, category) #path to images dir
    for img in os.listdir(path):
        img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE) #CHANGE THIS TO IMREAD_ANYCOLOUR
        # plt.imshow()
        break
    break

IMG_SIZE = 60
new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE)) #resizing images so they are the same size 
#plt.imshow(new_array, cmap = 'gray')
#plt.show()

training_data = []

def create_training_data():
    for category in CATEGORIES:
        path = os.path.join(DATADIR, category) #path to images dir
        class_num = CATEGORIES.index(category)
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_ANYCOLOR)
                new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
                training_data.append([new_array, class_num])
            except Exception as e:
                pass

create_training_data()

print(len(training_data))

random.shuffle(training_data)

for sample in training_data[:40]:
    print(sample[1])
    
X = []
y = []


for features, label in training_data:
    X.append(features)
    y.append(label) 
    
X = np.array(X) #X has to be a numpyarray # 1 should be 3 for color
y = np.array(y)

pickle_out = open("X.pickle", "wb")
pickle.dump(X, pickle_out)
pickle_out.close()

pickle_out = open("y.pickle", "wb")
pickle.dump(y, pickle_out)
pickle_out.close()