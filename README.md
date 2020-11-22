# fingers_counter
A very simple and interesting project where the number of fingers in a hand is detected, detecting also whether it is a right or left hand.

# Dataset 
is from kaggle you can find it here : https://www.kaggle.com/koryakinp/fingers 
18000 images is provided for training and 3600 images for testing.

# project codes
the project consists of three python files : 

preprocessing_modelTraining.py : which takes the dataset and perform some functions on it to resize all images to be (128 x 128), and extract the pictures labels from the last two bytes of the pic name. A CNN is used for training fiving an accuracy of 100%.

fingers_counter_model.py : two functions is used, the 1st one is to create the model and the 2nd one is to print the number of fingers.

test.py : takes any hand picture and input it to the model to predict the number of fingers.

# Model
A CNN is used with conv2d and maxpoolying layers, here is a summary for the model.

as shown a softmax activation function is used at the last dense layer to give a prediction to the number of hand's fingers.

