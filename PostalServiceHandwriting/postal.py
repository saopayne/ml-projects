# import the necessary libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv('zip-train.csv', sep=' ', header=None)
print(dataset.shape)
print(dataset.iloc[:, 0])
X_train = dataset.iloc[:, 1:255].values
y_train = dataset.iloc[:, 0].values

test_set = pd.read_csv('zip-test.csv', sep=' ', header=None)
X_test = test_set.iloc[:, 1:255].values
y_test = test_set.iloc[:, 0].values

len_train_set = len(dataset)
print("######The length of the training set ", len_train_set, " #######")
print("######The length of the test set ", len(test_set), " #######")

from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import r2_score

# Test out the test and train error using Linear Regression
regressor = LinearRegression()
regressor.fit(X_train, y_train)
y_reg_pred = regressor.predict(X_test)

# Linear regression score
print(r2_score(y_test, y_reg_pred))

# Test the test and train error using K nearest neighbor
classifier = KNeighborsClassifier(n_neighbors=15, metric='minkowski', p=2)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

print(y_reg_pred)
print(y_pred)

# KNN score
print(r2_score(y_test, y_pred))

# Visualising the Training set results
plt.scatter(X_train[:, 0], y_train, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Number vs Gray colour values (Training set)')
plt.xlabel('Gray codes')
plt.ylabel('Number')
plt.show()

# Visualising the Test set results
plt.scatter(X_test[:, 0], y_test, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Number vs Gray colour values (Test set)')
plt.xlabel('Gray codes')
plt.ylabel('Number')
plt.show()

