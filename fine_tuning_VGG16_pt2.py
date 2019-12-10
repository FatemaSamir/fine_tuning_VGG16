
#Implementation of the tutorial (for learning purposes):
#"Building powerful image classification models using very little data"
#https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html


import numpy as np
from keras import applications
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.preprocessing import image
from keras.utils import np_utils
from keras.models import Sequential, load_model
from keras.layers import Dense, Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img


#for plotting & image viewing
import scipy.misc
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
get_ipython().run_line_magic('matplotlib', 'inline')
from ipykernel import kernelapp as app
import imageio

#for time stamps
import datetime


#dimensions of the images
img_width, img_height = 150, 150


#weights path
top_model_weights_path = 'bottleneck_fc_model.h5'

#image directories
train_data_dir = 'train'
validation_data_dir = 'validation'
nb_train_samples = 2000
nb_validation_samples = 608
epochs = 50
batch_size = 16


def save_bottleneck_features():
    # this is the data generator we will use for training (no augmentation)
    datagen = ImageDataGenerator(rescale=1./255)
    
    #build the VGG16 network base
    model = applications.VGG16(include_top=False, weights='imagenet')
    
    # Train generator
    generator = datagen.flow_from_directory(
        train_data_dir, # this is the target directory
        target_size=(img_width, img_height), # all images will be resized to 150x150
        batch_size=batch_size,
        class_mode=None,  
        shuffle=False)  
    
    # the predict_generator method returns the output of a model, given
    # a generator that yields batches of numpy data
    print("Starting train bottleneck features", datetime.datetime.now())
    bottleneck_features_train = model.predict_generator(generator, nb_train_samples // batch_size)
    print("Train bottleneck features complete", datetime.datetime.now())


    # save the output as a Numpy array
    np.save(open('bottleneck_features_train.npy', 'wb'), bottleneck_features_train)
    print("Train bottleneck features saved", datetime.datetime.now())
    
    #Validation
    generator = datagen.flow_from_directory(
        validation_data_dir, 
        target_size=(img_width, img_height), # all images will be resized to 150x150
        batch_size=batch_size,
        class_mode=None,  
        shuffle=False)  
    
    # the predict_generator method returns the output of a model, given
    # a generator that yields batches of numpy data
    print("Starting validation bottleneck features", datetime.datetime.now())
    bottleneck_features_validation = model.predict_generator(generator, nb_validation_samples // batch_size)
    print("Validation bottleneck features complete", datetime.datetime.now())

    
    # save the output as a Numpy array
    np.save(open('bottleneck_features_validation.npy', 'wb'), bottleneck_features_validation)
    print("Validation bottleneck features saved", datetime.datetime.now())
    


def train_top_model():
    train_data = np.load(open('bottleneck_features_train.npy', 'rb'))
    
    # the features were saved in order, so recreating the labels is easy
    train_labels = np.array(
        [0] * (nb_train_samples // 2) + [1] * (nb_train_samples // 2))

    
    validation_data = np.load(open('bottleneck_features_validation.npy', 'rb'))
    validation_labels = np.array(
        [0] * (nb_validation_samples // 2) + [1] * (nb_validation_samples // 2))
       

    #Simple sequential model 
    model = Sequential()
    model.add(Flatten(input_shape=train_data.shape[1:]))  
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])
    
    model.fit(train_data, train_labels,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(validation_data, validation_labels))
    
    model.save_weights(top_model_weights_path)
    return model


save_bottleneck_features()



train_top_model()

