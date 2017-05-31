# import section

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# read in the values
dataset = pd.read_csv('risk_factors_cervical_cancer.csv')
X = dataset.iloc[:, 0:12].values
y = dataset.iloc[:, 28].values

# split the dataset into test and training set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.25, random_state= 0)

# scaling the feature
# scale the age
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.fit(X_test)

# get a classifier running
