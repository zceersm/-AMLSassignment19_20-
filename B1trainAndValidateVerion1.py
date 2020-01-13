from keras.layers import Dense, Conv2D, MaxPooling2D, BatchNormalization, GlobalAveragePooling2D
from keras import models
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

#name tf.variables_initializer is deprecated. Please use tf.compat.v1.variables_initializer instead.

# starting point 
ram_model= models.Sequential()

# 1st convolutional block
ram_model.add(Conv2D(24, (3, 3), activation='relu', padding='same', 
                    input_shape=(500,500,3)))
ram_model.add(MaxPooling2D((2, 2), padding='same'))
# 2nd block
ram_model.add(Conv2D(24, (3, 3), activation='relu', padding='same'))
ram_model.add(MaxPooling2D((2, 2), padding='same'))
# 3rd block
ram_model.add(Conv2D(48, (3, 3), activation='relu', padding='same'))
ram_model.add(MaxPooling2D((2, 2), padding='same'))
# 4th block
ram_model.add(Conv2D(96, (3, 3), activation='relu', padding='same'))
ram_model.add(MaxPooling2D((2, 2), padding='same'))
# global average pooling
ram_model.add(GlobalAveragePooling2D())
# completely connected layer
#ram_model.add(Dense(64, activation='relu'))
#ram_model.add(BatchNormalization())
#predicting
ram_model.add(Dense(5, activation='softmax'))


#use early stopping to stop training
early_stop=EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5)


# compile model 
ram_model.summary()
ram_model.compile(optimizer='adam', loss='categorical_crossentropy', 
                 metrics=['accuracy'])

#setting data generator up
Image_data_generator = ImageDataGenerator()

# get training images from my directory in batches
train_image_generator = Image_data_generator.flow_from_directory(
        'Datasets/B1/train',
        target_size=(500, 500),
        batch_size=50,
        class_mode='categorical', shuffle=True)

# get validation images from my directory in batches
validation_image_generator = Image_data_generator.flow_from_directory(
        'Datasets/B1/validate',
        target_size=(500, 500),
        batch_size=50,
        class_mode='categorical', shuffle=True)


history = ram_model.fit_generator(
        train_image_generator,
        epochs=10,
        steps_per_epoch=160,
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

    



