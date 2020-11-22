# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 23:00:53 2020

@author: Reham
"""

from skimage import io, transform
import glob
import numpy as np
import matplotlib.pyplot as plt

train_images = glob.glob("C:/Users/Reham/Desktop/fingers counter/fingers/train/*.png")
test_images = glob.glob("C:/Users/Reham/Desktop/fingers counter/fingers/test/*.png")

X_train = []
X_test = []
y_train = []
y_test = []

#===================Extract the number of fingers from the photo name====================    
for img in train_images:
    img_read = io.imread(img)
    # make sure all images are of size (128,128) 
    img_read = transform.resize(img_read, (128,128), mode = 'constant')
    X_train.append(img_read)
    # '00b4ea47-1329-4c8a-987c-15c39d259985_3R.png' the 3 is number of fingers and L is left hand so needed caracters are the two before.png
    y_train.append(img[-6:-4])
    
for img in test_images:
    img_read = io.imread(img)
    img_read = transform.resize(img_read, (128,128), mode = 'constant')
    X_test.append(img_read)
    y_test.append(img[-6:-4])
    
#(18000, 128, 128) (3600, 128, 128)
X_train = np.array(X_train)
X_test = np.array(X_test)


X_train = np.expand_dims(X_train, axis=3)
X_test = np.expand_dims(X_test, axis=3)
print('after expansion',X_train.shape,X_test.shape)  # prepare it as an input to the model(18000, 128, 128,1) (3600, 128, 128,1)


# convert labels to integers of y_train so the model can work properly
label_to_int={
    '0R' : 0,
    '1R' : 1,
    '2R' : 2,
    '3R' : 3,
    '4R' : 4,
    '5R' : 5,
    '0L' : 6,
    '1L' : 7,
    '2L' : 8,
    '3L' : 9,
    '4L' : 10,
    '5L' : 11
}

temp = []
for label in y_train:
    temp.append(label_to_int[label])
y_train = temp.copy()

temp = []
for label in y_test:
    temp.append(label_to_int[label])
y_test = temp.copy()


#========================model==========================================================

import tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D

y_train = tensorflow.keras.utils.to_categorical(y_train, num_classes = 12)
y_test = tensorflow.keras.utils.to_categorical(y_test, num_classes = 12)

model = Sequential()
model.add(Conv2D(32, (3,3), input_shape = (128, 128, 1), activation = 'relu'))
model.add(Conv2D(32, (3,3), activation = 'relu'))

model.add(Conv2D(64, (3,3), activation = 'relu'))
model.add(Conv2D(64, (3,3), activation = 'relu'))

model.add(MaxPooling2D((2,2)))

model.add(Conv2D(128, (3,3), activation = 'relu'))
model.add(Conv2D(128, (3,3), activation = 'relu'))

model.add(Flatten())

model.add(Dropout(0.40))
model.add(Dense(256, activation = 'relu'))
model.add(Dropout(0.40))
model.add(Dense(12, activation = 'softmax'))

model.compile('SGD', loss = 'categorical_crossentropy', metrics = ['accuracy'])
model.fit(x = X_train, y = y_train, batch_size = 128, epochs = 10, validation_data = (X_test, y_test))

model.save('fingers_counter.h5')

pred = model.evaluate(X_test,
                      y_test,
                    batch_size = 128)

print("Accuracy of model on test data is: ",pred[1]*100)
    