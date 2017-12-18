# Data Preprocessing

# Importing the essential libraries
# Name of library is numpy and alias is np
# numpy is a library that contains mathematical tools.
import numpy as np
# matplotlib is the library and pyplot is the sublibrary
# this library allows us to plot graphs in python
import matplotlib.pyplot as plt
# pandas is the library to import, manage and export datasets
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 3].values

# Taking care of missing data
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])