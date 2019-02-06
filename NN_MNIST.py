### MNIST Digit Recognition ###

############################ Load Libraries ###################################

import os
import time
import numpy as np
import random as rd
import tensorflow as tf
from tensorflow import keras
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import model_from_json
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

##################### Define my functions #####################################

### Plot the training of the NN
def plot_history(history):
  plt.figure()
  plt.title('Training History: Feed Forward Neural Net')
  plt.xlabel('Epoch')
  plt.ylabel('Error')
  plt.plot(history.epoch, 1-np.array(history.history['acc']),
           label='Train Error')
  plt.plot(history.epoch, 1-np.array(history.history['val_acc']),
           label = 'Validation Error')
  plt.legend()
  #plt.ylim([800, 1200])

### Get class-specific error rates from a confusion matrix
# Note on conf mat format: row=pred and col=actual

def class_err(conf_mat):
    conf_mat = np.array(conf_mat) # convert to np array
    class_count = conf_mat.shape[0]
    error_vec = np.zeros(class_count)
    I = range(conf_mat.shape[0])
    
    for i in I:
        no_correct = conf_mat[i,i]
        no_class_obs = np.sum(conf_mat, axis=0)[i]
        error_vec[i] = 1 - no_correct / no_class_obs

    print('\nClass-Specific Error Rates:\n')
    for i in I:
        print('Class = %d: %.2f' %(i,round(error_vec[i],4)*100),'%')
  
# Visualise the MNIST data
def VIEW(index, data):
    image = data[index]
    image = np.array(image, dtype='float')
    pixels = image.reshape((28, 28))
    plt.imshow(pixels, cmap='gray')
    plt.show()
    
# Find indices for test predictions given predicted value and actual value
def IndFind(pred_val, actual_val, pred_vec, actual_vec):
    indices = []
    for i in range(len(pred_vec)):
        if (pred_vec[i]==pred_val) and (actual_vec[i]==actual_val):
            indices.append(i)
    return indices
    
############################ Load the MNIST Data and create splits ############

from keras.datasets import mnist

# Train test split
(x_train, y_train), (x_test, y_test) = mnist.load_data()
y_test_classes = y_test # used later to generate a confusion matrix

# Convert each image matrix to a vector
vec_len = x_test.shape[1] * x_test.shape[2]
x_train = x_train.reshape(x_train.shape[0], vec_len)
x_test = x_test.reshape(x_test.shape[0], vec_len)

# Create the validation set (20% of test set)
x_train, x_valid, y_train, y_valid = train_test_split(x_train, 
                                                      y_train, 
                                                      test_size=0.2, 
                                                      random_state=1)

# Convert the y-values to dummy matrices
class_count = 10
y_train = keras.utils.to_categorical(y_train, class_count)
y_valid = keras.utils.to_categorical(y_valid, class_count)
y_test = keras.utils.to_categorical(y_test, class_count)

##################### Create the Feed Forward Neural Net Model ################
np.random.seed(1)
model = keras.Sequential()
model.add(keras.layers.Dense(units=500, input_dim=x_train.shape[1], 
                             activation='relu'))
model.add(keras.layers.Dense(units=100, activation='relu'))
model.add(keras.layers.Dense(units=class_count, activation='softmax'))
model.summary()

##################### Define optimiser and compile the model ##################

optimizer = keras.optimizers.Adam(lr=0.001)

early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=20)

model.compile(optimizer=optimizer, 
              loss='categorical_crossentropy', 
              metrics=['accuracy'])

train_time_0 = time.time()

history = model.fit(x_train, y_train, 
                    epochs=500, 
                    batch_size=1000, 
                    validation_data = (x_valid, y_valid),
                    callbacks=[early_stop])

train_time_1 = time.time()
train_time = train_time_1 - train_time_0
print('Training time = %s minutes' % round(train_time/60,2))

# Plot the training and validation error
plot_history(history)

##################### Evaluate on the validation set ##########################

val_loss, val_accuracy = model.evaluate(x_valid, y_valid)
print('Validation Error = ',round(1-val_accuracy,4)*100,'%')

##################### Evaluate on the test set ################################

test_loss, test_accuracy = model.evaluate(x_test, y_test)
print('Test Error = ',round(1-test_accuracy,4)*100,'%')

# Generate class predictions (0 to 9)
test_preds = model.predict_classes(x_test)

# Create a confusion matrix
conf_mat = confusion_matrix(test_preds, y_test_classes)
conf_mat = pd.DataFrame(conf_mat)
print('Confusion Matrix: (rows=predicted, columns=actual)\n\n',conf_mat)

# Return the class-specific error rates
class_err(conf_mat)

##################### View incorrectly classified images ######################

# Get error indices
errors = np.array(np.nonzero(test_preds != y_test_classes))

# View the images for given mis-classification and actual values
pred_val = 0
actual = 8
for i in IndFind(pred_val, actual, test_preds, y_test_classes):
    VIEW(i, x_test)

##################### Save the model ##########################################

## Set WD
#os.chdir('')
## Save Model
#model_json=model.to_json()
#with open("NN_model_MNIST.json","w") as json_file:
#    json_file.write(model_json)
#model.save_weights("NN_model_MNIST.h5") # serialise weights to HDF5

##################### Load a model ############################################

#json_file = open('NN_model_MNIST.json', 'r')
#loaded_model_json = json_file.read()
#json_file.close()
#loaded_model = model_from_json(loaded_model_json)
## load weights into new model
#loaded_model.load_weights("NN_model_MNIST.h5")
#print("Model has been loaded")
#
## Compile the model
#loaded_model.compile(optimizer=optimizer, 
#              loss='categorical_crossentropy', 
#              metrics=['accuracy'])

