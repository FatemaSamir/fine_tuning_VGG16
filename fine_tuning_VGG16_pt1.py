
# coding: utf-8

#Implementation of the tutorial:
#"Building powerful image classification models using very little data"
#https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html


import numpy as np
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.preprocessing import image
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K

#for plotting & image viewing
import scipy.misc
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
get_ipython().run_line_magic('matplotlib', 'inline')
from ipykernel import kernelapp as app
import imageio


#data preprocessing example, to illustrate image transformations
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')


img = load_img('train/calico/calico1.jpg')  # this is a PIL image
x = img_to_array(img)  # this is a Numpy array with shape (3, 150, 150)
x = x.reshape((1,) + x.shape)  # this is a Numpy array with shape (1, 3, 150, 150)

# the .flow() command below generates batches of randomly transformed images
# and saves the results to the `preview/` directory
i = 0
for batch in datagen.flow(x, batch_size=1,
                          save_to_dir='preview', save_prefix='calico', save_format='jpeg'):
    i += 1
    if i > 20:
        break  # otherwise the generator would loop indefinitely

#dimensions of the images 
img_width, img_height = 150, 150


#image directories
train_data_dir = 'train'
validation_data_dir = 'validation'
nb_train_samples = 2000
nb_validation_samples = 600
epochs = 50
batch_size = 16

#channels ordering
if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)


#Build a simple Sequential CNN model
model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=input_shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# the model so far outputs 3D feature maps (height, width, features)

model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])


# this is the data augmentation to be used for training
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

# this is the augmentation configuration to be used for testing:
# only rescaling
test_datagen = ImageDataGenerator(rescale=1./255)

# this is a generator that will read pictures found in
# subfolers of 'train', and indefinitely generate
# batches of augmented image data
train_generator = train_datagen.flow_from_directory(
        train_data_dir,  # this is the target directory
        target_size=(img_width, img_height),  # all images will be resized to 150x150
        batch_size=batch_size,
        class_mode='binary')  # since we use binary_crossentropy loss, we need binary labels

# this is a similar generator, for validation data
validation_generator = test_datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_width, img_height),
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
model.save_weights('weights/30_epochs.h5')  # always save your weights after training or during training


#load and pre-process a new image to make predictions
img_path = 'shrimp.jpg'
img = image.load_img(img_path, target_size=(150, 150))
x = image.img_to_array(img)
x = x / 255.0
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



#load and pre-process another new image to make predictions
img_path = 'mochi.jpg'
img = image.load_img(img_path, target_size=(150, 150))
x = image.img_to_array(img)
x = x / 255.0
x = np.expand_dims(x, axis=0)

my_image = imageio.imread(img_path)
imshow(my_image)

#make predictions using the trained model
prob = model.predict(x)
preds = model.predict_classes(x) 

for catclass, label in train_label_map.items():
    if preds == label:
        print("Class label:", preds, "Class name:", catclass)
              
print("Probability:", prob)    

