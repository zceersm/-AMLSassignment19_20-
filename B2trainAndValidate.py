from numpy import loadtxt,argmax,zeros,sum, set_printoptions; from numpy.random import shuffle
from keras.models import Sequential;from keras.layers import Dense, Dropout, LeakyReLU
from keras.utils import to_categorical; import argparse
from keras.layers import Dense, Conv2D, MaxPooling2D, BatchNormalization, GlobalAveragePooling2D
from keras import models
import matplotlib.pyplot as plt

csvFile = "AMLS/B2/eyeData.csv"

def loadData():
	dataset = loadtxt(csvFile, delimiter=',');shuffle(dataset)
	X = dataset[:,1:];rawY = dataset[:,0];Y=to_categorical(rawY)
	numberOfRows = X.shape[0]; numberOfInputs = X.shape[1]; numberOfClasses=Y.shape[1]
	return X,Y,numberOfRows,numberOfInputs,numberOfClasses

X,Y,numberOfRows,numberOfInputs,numberOfClasses = loadData()
model = Sequential()
model.add(Dense(200, input_dim=numberOfInputs, activation='relu'))
model.add(Dense(150, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(numberOfClasses, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

history = model.fit(X, Y, epochs=15, batch_size=16,validation_split=0.2)
loss, accuracy = model.evaluate(X, Y)
                  
model.save('model.h5')
model.summary()

# list all data in history
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.ylim((0,1))
plt.legend(['train', 'validate'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['train', 'validate'], loc='upper left')
plt.show()