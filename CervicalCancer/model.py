"""
This is a sample neural network built with Keras using Tensorflow as a backend
Code is written in Python 3
This neural network goes through a dataset on cervical cancer patients
Written by: saopayne
Source for dataset:
https://archive.ics.uci.edu/ml/datasets/Cervical+cancer+%28Risk+Factors%29
"""

# SUPPRESS ALL WARNINGS:
import warnings
warnings.filterwarnings("ignore")

# IMPORT MODUELS

import csv              # Used to read CSV files

import time             # Keep track of time

import matplotlib.pyplot as plt     # Generating 2D charts
import pandas as pd
import keras            # Used for neural network
from keras.models import Sequential
from keras.layers import Dense
from keras import optimizers        # Used for optimization

import numpy as np      # Used to analyze data and arrays

# DECLARE VARIABLES

master_array = []       # Stores all the raw data
master_array_2 = []     # Stores all the treated data

X = []                  # Untreated data
Y = []                  # Results

start = time.time()
# Description of the program
print("Sample Neural Net Used on Patient Data for Cervical Cancer")


# Read CSV file function
def read_file(file_name):
    with open(file_name) as csvfile:
        readCSV = csv.reader(csvfile, delimiter = ',')  # Delimits
        for row in readCSV:
            master_array.append(row)    # Appends to master_array


# Remove rows with '?' function
def remove_symbol(symbol, array_pr, new_array):
    for row in array_pr:
        if symbol in row:
            array_pr.remove(row)    # Removes rows with list
        else:
            new_array.append(row)


# Process array function
def process_array(array_pr):
    for row in array_pr:        # For each row
        array_pr = [float(item) for item in row]


def list_size_2dim(array_pr):
    print(len(array_pr))

# Load dataset
read_file('risk_factors_cervical_cancer.csv')
print("Reading file...")

# Remove symbol
remove_symbol('?', master_array, master_array_2)
print("Processing data")

# Process array
del master_array_2[:2]    # Removes first 2 rows

process_array(master_array_2)
print("Converting data")

print("Number of items:")
print(len(master_array_2))

# Assign data to lists

# Y: Results
for row in master_array_2:
    row = [float(x) for x in row]
    Y.append(row[26])

# X: Data
for row in master_array_2:
    row = [float(x) for x in row]
    del row[26:]
    X.append(row)

print("Data ready \n")


# Build neural network
# Sequential
model = Sequential()

# Neural network
model.add(Dense(20, input_dim=26, init='uniform', activation='sigmoid' ))
model.add(Dense(25, init='uniform', activation='sigmoid'))
model.add(Dense(25, init='uniform', activation='sigmoid'))
model.add(Dense(20, init='uniform', activation='sigmoid'))
model.add(Dense(8, init='uniform', activation='sigmoid'))
model.add(Dense(1, init='uniform', activation='sigmoid'))


# Compile model
sgd = optimizers.SGD(lr=0.0001, decay=1e-10, momentum=0.01, nesterov=True)
model.compile(loss='mean_squared_logarithmic_error', optimizer='SGD', metrics=['accuracy'])

# Fit model
history = model.fit(X, Y, nb_epoch=200, validation_split=0.1, batch_size=10)

end = time.time()

elapsed = end - start

print("ELAPSED TIME:", elapsed)

# Analysis and data plot

plt.figure(1)
plt.subplot(211)
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy (%)')
plt.xlabel("Epoch")
plt.legend(['train', 'test'], loc='lower right')

# summarize history for loss
plt.subplot(212)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss (%)')
plt.xlabel("Epoch")
plt.legend(['train', 'test'], loc='lower left')
plt.show()
print('jh')