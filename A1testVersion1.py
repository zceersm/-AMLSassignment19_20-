
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

    



