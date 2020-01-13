from keras.layers import Dense, Conv2D, MaxPooling2D, BatchNormalization, GlobalAveragePooling2D
from keras import models
from tensorflow.python.keras.applications.vgg16 import preprocess_input
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, ModelCheckpoint

#name tf.variables_initializer is deprecated. Please use tf.compat.v1.variables_initializer instead.

# starting point 
ram_model= models.Sequential()

# 1st convolutional block
ram_model.add(Conv2D(16, (3, 3), activation='relu', padding='same', 
                    input_shape=(178,218,3))) #this convolution layer uses 16, (3 by 3) pixel filters that are applied to each part of the image, returning 16 arrays of activation values called feature maps that indicate where certain features are located in the image.
ram_model.add(MaxPooling2D((2, 2), padding='same'))

# 2nd block
ram_model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
ram_model.add(MaxPooling2D((2, 2), padding='same'))
# 3rd block
ram_model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
ram_model.add(MaxPooling2D((2, 2), padding='same'))
# 4th block
ram_model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
ram_model.add(MaxPooling2D((2, 2), padding='same'))

# global average pooling
ram_model.add(GlobalAveragePooling2D())
# completely connected layer
ram_model.add(Dense(64, activation='relu'))
ram_model.add(BatchNormalization())
#predicting
ram_model.add(Dense(2, activation='sigmoid'))


#use early stopping to stop training
early_stop=EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5)

# saving the best model 
model_check_point = ModelCheckpoint("C:\\Users\\faymi\\Desktop\\4th_year_Stuff\\ML\\dataset_AMLS_19-20\\ram_model.h5", monitor='val_loss', mode='min', verbose=1, save_best_only = True)
call_back_list = [early_stop,model_check_point]


# compile model 
ram_model.compile(optimizer='adam', loss='binary_crossentropy', 
                 metrics=['accuracy'])




#setting data generator up
Image_data_generator = ImageDataGenerator(preprocessing_function=preprocess_input)

# get training images from my directory in batches
train_image_generator = Image_data_generator.flow_from_directory(
        'C:\\Users\\faymi\\Desktop\\4th_year_Stuff\\ML\\dataset_AMLS_19-20\\RamasuDatasets\\A1\\training',
        target_size=(178, 218),
        batch_size=32,
        class_mode='categorical')

# get validation images from my directory in batches
validation_image_generator = Image_data_generator.flow_from_directory(
        'C:\\Users\\faymi\\Desktop\\4th_year_Stuff\\ML\\dataset_AMLS_19-20\\RamasuDatasets\\A1\\validation',
        target_size=(178, 218),
        batch_size=32,
        class_mode='categorical')


history = ram_model.fit_generator(
        train_image_generator,
        epochs=3,
        steps_per_epoch=2667,
        validation_data=validation_image_generator,
        validation_steps=667, callbacks=call_back_list)



# plotting accuracy of training and validation
import matplotlib.pyplot as plt
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.ylim([.5,1.1])
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.savefig("Accuracy Of Training And Validation.png", dpi=300)


#******Testing *********************

# load saved model
from keras.models import load_model
import os
os.chdir('C:\\Users\\faymi\\Desktop\\4th_year_Stuff\\ML\\dataset_AMLS_19-20\\RamasuDatasets\\A1\\')
saved_model = load_model('Custom_Keras_CNN.h5')

# generating data for set of test images
test_generator = Image_data_generator.flow_from_directory(
        'C:\\Users\\faymi\\Desktop\\4th_year_Stuff\\ML\\dataset_AMLS_19-20\\RamasuDatasets\\A1\\testing',
        target_size=(178, 218),
        batch_size=10,
        class_mode='categorical',
        shuffle=False)

# here we get the predicted activation values of the last dense layer
import numpy as np
test_generator.reset()
predict=saved_model.predict_generator(test_generator, verbose=1, steps=1000)

# for each sample we determine the maximum activation values 
predicted_class_indices=np.argmax(predict,axis=1)

# assign the predicted value to appropriate gender
labels = (test_generator.class_indices)
labels = dict((v,k) for k,v in labels.items())
predictions = [labels[k] for k in predicted_class_indices]

# change names of files to either male or female
filenames=test_generator.filenames
filesnz=[0]
for i in range(0,len(filenames)):
    filesnz.append(filenames[i].split('\\')[0])
filesnz=filesnz[1:]

# ascertain  accuracy of test set
match=[]
for i in range(0,len(filenames)):
    match.append(filesnz[i]==predictions[i])
match.count(True)/1000


    



    
    
    

def buildNN():
    if verbose: print("Building Neural Network")
    ram_model = models.Sequential()
    ram_model.add(Conv2D(16, (3, 3), activation='relu', padding='same', input_shape=numberOfInputs)) 
    ram_model.add(MaxPooling2D((2, 2), padding='same'))
    ram_model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
    ram_model.add(MaxPooling2D((2, 2), padding='same'))
    ram_model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    ram_model.add(MaxPooling2D((2, 2), padding='same'))
    ram_model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    ram_model.add(MaxPooling2D((2, 2), padding='same'))
    ram_model.add(GlobalAveragePooling2D())
    ram_model.add(Dense(64, activation='relu'))
    ram_model.add(BatchNormalization())
    ram_model.add(Dense(5, activation='softmax'))
    ram_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return(ram_model)

def trainNN():
	if verbose: print("Training Neural Network")
	ram_model.fit(X, Y, epochs=10, batch_size=16,validation_split=0.2, verbose=verbose)
	loss, accuracy = ram_model.evaluate(X, Y, verbose=verbose)
	return(ram_model,accuracy)

def save(ram_model):
	ram_model.save(modelFile)
    
X,Y,numberOfRows,numberOfInputs,numberOfClasses = loadData()
ram_model = buildNN()
ram_model,trainAccuracy = trainNN()
print("Training complete. Accuracy: %2.1f%%"%(trainAccuracy*100))
saveModel(ram_model)

