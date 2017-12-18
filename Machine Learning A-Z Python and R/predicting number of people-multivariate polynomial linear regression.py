# DATA PREPROCESSING STEPS
# 1. Import Essential Libraries
# 2. Import the Dataset
# 3. Split the Dataset into 2 matrices X (independent variables) & y(dependent variable)
# 4. Replace missing data with the mean
# 5. Encoding categorial data (only if there is categorical data)
# 6. Avoid dummy variable (only if there is categorical data)
# 7. Split(randomly) the dataset into training and test set
# 8. Feature Scaling
# DATA ANALYTICS MODEL BUILDING
# 9. Fitting multivariate linear regression to training set
# 10. Predict test set results
# 11. Calculate std. deviation
# 12. Optimize the model using backward elimination (iterative steps)
# 13. Use optimized model, remove columns that are not needed and repeat steps 9 -12
#
# IMPORTING ESSENTIAL LIBRARIES
## Name of library is numpy and alias is np
## numpy is a library that contains mathematical tools.
import numpy as np
## matplotlib is the library and pyplot is the sublibrary
## this library allows us to plot graphs in python
import matplotlib.pyplot as plt
## pandas is the library to import, manage and export datasets
import pandas as pd

# IMPORTING THE DATASET
## Set the working directory to the source of the csv file
dataset = pd.read_csv('Data.csv')
## Format of input file contains Temp, Humidity, Luminosity, Motion, No. of people in the room
## Create a matrix of Temp, Humidity, Luminosity and Motion
## X is matrix of independent variables
X = dataset.iloc[:, :-1].values
## Create a vector of no. of people
## y is a dependent variable for no. of people in the room
y = dataset.iloc[:, 4].values

# TAKING CARE OF MISSING DATA (adding mean of column) for missing observations
# Another option is to delete the observation but taking mean is recommended over deleting an observation
## sklearn is scikit library that contains machine learning models
## preprocessing is a sublibrary that contains methods for preprocessing data
from sklearn.preprocessing import Imputer
## The below step will replace all missing values'nan' in the csv file with mean(strategy) of column(axis) 
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
## The next step fits the imputer object into our matrix X.
imputer = imputer.fit(X[:, 0:4])
## replace the missing data of matrix X with mean of column
X[:, 0:4] = imputer.transform(X[:, 0:4])

## ENCODING CATEGORICAL DATA - Converting named categories into numerical values
# from sklearn.preprocessing import LabelEncoder, OneHotEncoder
# labelencoder_X = LabelEncoder()
# X[:, 0] = labelencoder_X.fit_transform(X[:, 0])
#
# labelencoder_y = LabelEncoder()
# y = labelencoder_y.fit_transform(y)
#
## Creating dummy variables for countries
# onehotencoder_X = OneHotEncoder(categorical_features = [0])
# X = onehotencoder_X.fit_transform(X).toarray()
#
# Adding a column of ones to avoid dummy variables
# X = X[:, 1:]

# SPLITTING THE DATASET INTO TRAINING(70%) AND TEST SET(30%)
## cross_validation is a sublibrary in sklearn that contains methods to split the dataset
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)

# FEATURE SCALING
## StandardScaler is a method that scales the features using standard deviation
## for every element in column x = (x - mean(x))/standard deviation(x)
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
"""sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""

# FITTING MULTIVARIATE LINEAR REGRESSION TO THE TRAINING SET
# Fitting Simple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Fitting Polynomial Regression to the dataset
# modifying X to add columns for quadratic variables square(x1), square(x2), square(x3), square(x4), x1*x2, x1*x3, x1*x4, x2*x3, x2*x4, x3*x4
from sklearn.preprocessing import PolynomialFeatures
poly_regressor = PolynomialFeatures(degree = 2)
X_poly = poly_regressor.fit_transform(X)
X_train_poly = poly_regressor.fit_transform(X_train)
X_test_poly = poly_regressor.fit_transform(X_test)
regressor2 = LinearRegression()
regressor2.fit(X_train_poly, y_train)

# Predicting the Test set results
y_pred = regressor.predict(X_test)
y_pred_poly = regressor2.predict(X_test_poly)

# Calculating Standard Deviation
std_dev = np.sum(np.square(y_pred - y_test))/y_pred.shape[0]
std_dev_poly = np.sum(np.square(y_pred_poly - y_test))/y_pred.shape[0]
# I stopped here as the standard devistion increased as compared to the linear model.

# BUILDING OPTIMAL MODEL USING BACKWARD ELIMINATION
# Getting the number of rows in X, X.shape[0]
# appending 1's to X for constant in the equation
import statsmodels.formula.api as sm
X = np.append(arr = np.ones((X.shape[0], 1)).astype(int), values = X, axis = 1)

X_opt = X[:, [0, 1, 2, 3, 4]] # specifying the indexes of each column
# Fit the full model with all possible predictors
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()

X_opt = X[:, [0, 1, 2, 3]] # specifying the indexes of each column
# Fit the full model with all possible predictors
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()

"""# Visualising the Training set results
plt.scatter(X_train, y_train, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Salary vs Experience (Training set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

# Visualising the Test set results
plt.scatter(X_test, y_test, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Salary vs Experience (Test set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()"""