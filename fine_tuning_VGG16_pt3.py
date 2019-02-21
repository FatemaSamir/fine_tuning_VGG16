
# coding: utf-8

#Implementation of the tutorial:
#"Building powerful image classification models using very little data"
#https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html

import numpy as np
from keras import applications
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.preprocessing import image
from keras.utils import np_utils
from keras.models import Sequential, load_model, Model
from keras.layers import Dense, Conv2D, MaxPooling2D, Input
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras import optimizers

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
weights_path = 'vgg16_weights.h5'
#top model weights path
top_model_weights_path = 'bottleneck_fc_model.h5'

#image directories
train_data_dir = 'train'
validation_data_dir = 'validation'
nb_train_samples = 2000
nb_validation_samples = 608
epochs = 50
batch_size = 16

#channels ordering
if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)


#build VGG network base
input_tensor = Input(shape=input_shape)
base_model = VGG16(weights='imagenet', include_top=False, input_tensor=input_tensor)
print('Model loaded.')

#build a classifier model to put on top of the CNN
top_model = Sequential()
top_model.add(Flatten(input_shape=base_model.output_shape[1:]))  
top_model.add(Dense(256, activation='relu'))
top_model.add(Dropout(0.5))
top_model.add(Dense(1, activation='sigmoid'))

# note that it is necessary to start with a fully-trained
# classifier, including the top classifier,
# in order to successfully do fine-tuning
top_model.load_weights(top_model_weights_path)

#These next two steps are to keep the model sequential
# copy all the layers of VGG16 to model
model = Sequential()
for l in base_model.layers:
    model.add(l)

# concatenate VGG16 and top model 
model.add(top_model)

# set the first 15 layers (up to the last conv block)
# to non-trainable (weights will not be updated)
for layer in model.layers[:15]:
    layer.trainable=False
    
#compile the model with SGD/momentum optimizer
#and a very slow learning rate
model.compile(loss='binary_crossentropy',
              optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
              metrics=['accuracy'])


# prepare the train data augmentation configuration
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

# prepare the test data no augmentation
test_datagen = ImageDataGenerator(rescale=1. / 255)

# train generator
train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary')

#validation generator
validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary')    


#this gives us a map of the class names
#here we have 2 classes for binary classification
train_label_map = (train_generator.class_indices)
print(train_label_map)
val_label_map = (validation_generator.class_indices)
print(val_label_map)


#train the model
model.fit_generator(
    train_generator,
    steps_per_epoch=nb_train_samples // batch_size,
    epochs=30,
    validation_data=validation_generator,
    validation_steps=nb_validation_samples // batch_size)
model.save_weights('fine_tune_30_epochs_seqmodel.h5')



#load and pre-process a new image to make predictions
img_path = 'shrimp.jpg'
img = image.load_img(img_path, target_size=(150, 150))
x = image.img_to_array(img)
x = x / 255.0 #rescale
x = np.expand_dims(x, axis=0)
my_image = imageio.imread(img_path)
imshow(my_image)

#make predictions using the trained model
prob = model.predict(x)
preds = model.predict_classes(x) 
 

for catclass, label in train_label_map.items():
    if preds == label:
        print("Class label:", preds, "Class name:", catclass)
              
print("Probability:", prob) #print the probability    




#load and pre-process a new image to make predictions
img_path = 'mochi.jpg'
img = image.load_img(img_path, target_size=(150, 150))
x = image.img_to_array(img)
x = x / 255.0
x = np.expand_dims(x, axis=0)

#make predictions using the trained model
my_image = imageio.imread(img_path)
imshow(my_image)

prob = model.predict(x)
preds = model.predict_classes(x) 

for catclass, label in train_label_map.items():
    if preds == label:
        print("Class label:", preds, "Class name:", catclass)
              
print("Probability:", prob) #print the probability 

