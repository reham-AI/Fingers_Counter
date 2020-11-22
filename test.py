# -*- coding: utf-8 -*-
"""
Created on Thu Nov 19 00:09:06 2020

@author: Reham
"""
import cv2
from tensorflow.keras.models import load_model
from fingers_counter_model import create_model
from fingers_counter_model import print_fingers


model_weights = 'fingers_counter_epoch10.h5'  # Trained model weights

model = create_model()

model.load_weights(model_weights)

curr_image= cv2.imread('reham_hand.jpg')

gray = cv2.cvtColor(curr_image, cv2.COLOR_BGR2GRAY)

gray = cv2.resize(gray,(128,128))

gray = gray.reshape(1,128,128,1)

pred_fingers = model.predict(gray)

print_fingers(pred_fingers)

