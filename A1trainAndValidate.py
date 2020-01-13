from keras.layers import Dense, Conv2D, MaxPooling2D, BatchNormalization, GlobalAveragePooling2D
from keras import models
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

#name tf.variables_initializer is deprecated. Please use tf.compat.v1.variables_initializer instead.

# starting point 
ram_model= models.Sequential()

# 1st convolutional block
ram_model.add(Conv2D(16, (3, 3), activation='relu', padding='same', 
                    input_shape=(218,178,3)))
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
ram_model.add(Dense(1, activation='sigmoid'))


#use early stopping to stop training
early_stop=EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5)


# compile model 
ram_model.summary()
ram_model.compile(optimizer='adam', loss='binary_crossentropy', 
                 metrics=['accuracy'])

#setting data generator up
Image_data_generator = ImageDataGenerator()

# get training images from my directory in batches
train_image_generator = Image_data_generator.flow_from_directory(
        'Datasets/A1/train',
        target_size=(218, 178),
        batch_size=10,
        class_mode='binary', shuffle=True)

# get validation images from my directory in batches
validation_image_generator = Image_data_generator.flow_from_directory(
        'Datasets/A1/validate',
        target_size=(218, 178),
        batch_size=10,
        class_mode='binary', shuffle=True)


history = ram_model.fit_generator(
        train_image_generator,
        epochs=40,
        steps_per_epoch=100,
        validation_data=validation_image_generator)#, callbacks=call_back_list)
ram_model.save("model.h5")


# list all data in history
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.ylim(0,1)
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

"""
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
"""

    



