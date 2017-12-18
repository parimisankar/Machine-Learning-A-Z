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

# Predicting the Test set results
y_pred = regressor.predict(X_test)

# Calculating Standard Deviation
std_dev = np.sum(np.square(y_pred - y_test))/y_pred.shape[0]
# With my data, I got std_dev = 3.55 (which means model will predict +/- 3.55 people in the room, Not acceptable)
# If you are satisfied with accuracy level, you can stop here
# Else, Implement Backward Elimination method to remove insignificant variables

# BUILDING OPTIMAL MODEL USING BACKWARD ELIMINATION
# Getting the number of rows in X, X.shape[0]
# appending 1's to X for constant in the equation
import statsmodels.formula.api as sm
X = np.append(arr = np.ones((X.shape[0], 1)).astype(int), values = X, axis = 1)

X_opt = X[:, [0, 1, 2, 3, 4]] # specifying the indexes of each column
# Fit the full model with all possible predictors
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()
# You will see
##################################################################################
#                             OLS Regression Results                             #
# ============================================================================== #
# Dep. Variable:                      y   R-squared:                       0.837 #
# Model:                            OLS   Adj. R-squared:                  0.707 #
# Method:                 Least Squares   F-statistic:                     6.430 #
# Date:                Mon, 18 Dec 2017   Prob (F-statistic):             0.0331 #
# Time:                        13:55:11   Log-Likelihood:                -9.9850 #
# No. Observations:                  10   AIC:                             29.97 #
# Df Residuals:                       5   BIC:                             31.48 #
# Df Model:                           4                                          #
# Covariance Type:            nonrobust                                          #
# ============================================================================== #
#                  coef    std err          t      P>|t|      [0.025      0.975] #
# ------------------------------------------------------------------------------ #
# const         17.3572      5.158      3.365      0.020       4.098      30.616 #
# x1            -0.6993      0.257     -2.725      0.042      -1.359      -0.040 #
# x2             0.0002   7.23e-05      2.431      0.059   -1.01e-05       0.000 #
# x3            -0.2479      0.106     -2.341      0.066      -0.520       0.024 #
# x4            -0.9313      0.948     -0.982      0.371      -3.369       1.506 #
# ============================================================================== #
# Omnibus:                        0.076   Durbin-Watson:                   1.761 #
# Prob(Omnibus):                  0.963   Jarque-Bera (JB):                0.164 #
# Skew:                          -0.126   Prob(JB):                        0.921 #
# Kurtosis:                       2.426   Cond. No.                     1.15e+06 #
# ============================================================================== #
##################################################################################
 
# P>|t| is called P-value and gives the significance level of the variable in the equation.
# Higher the p-value, lower the significance
# Remove the column with highest P>|t| value in the summary if P>|t| is > 0.05
# In this case, it is x4 so, remove it by not inputing the index

X_opt = X[:, [0, 1, 2, 3]] # specifying the indexes of each column
# Fit the full model with all possible predictors
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()
# You will see
##################################################################################
#                             OLS Regression Results                             #
# ============================================================================== #
# Dep. Variable:                      y   R-squared:                       0.806 #
# Model:                            OLS   Adj. R-squared:                  0.709 #
# Method:                 Least Squares   F-statistic:                     8.301 #
# Date:                Mon, 18 Dec 2017   Prob (F-statistic):             0.0148 #
# Time:                        13:56:05   Log-Likelihood:                -10.867 #
# No. Observations:                  10   AIC:                             29.73 #
# Df Residuals:                       6   BIC:                             30.94 #
# Df Model:                           3                                          #
# Covariance Type:            nonrobust                                          #
# ============================================================================== #
#                  coef    std err          t      P>|t|      [0.025      0.975] #
# ------------------------------------------------------------------------------ #
# const         20.6940      3.869      5.348      0.002      11.226      30.162 #
# x1            -0.8462      0.208     -4.073      0.007      -1.355      -0.338 #
# x2             0.0002   6.99e-05      2.266      0.064   -1.26e-05       0.000 #
# x3            -0.2323      0.104     -2.225      0.068      -0.488       0.023 #
# ============================================================================== #
# Omnibus:                        0.266   Durbin-Watson:                   2.132 #
# Prob(Omnibus):                  0.876   Jarque-Bera (JB):                0.130 #
# Skew:                          -0.191   Prob(JB):                        0.937 #
# Kurtosis:                       2.592   Cond. No.                     8.56e+05 #
# ============================================================================== #
##################################################################################
# Again look at P>|t| and remove the row with highest value that is > 0.05 
# (In this case, I ignored as it is only slightly above 0.05)
# Other variables of interest in the above table are coef, R-Squared and Adj. R-squared
# Coef are coefficients in the prediction equation
# R-squared is goodness of the model's fit.
# R-squared and adjusted R-squared are normally between 0 and 1
# The closer these R-squared and Adj. R-squared values are to 1, the better the model is.
# The prediction equation will be number of people (y) = coef(const) + coef(x1)*x1 + coef(x2)*x2 + coef(x3)*x3
# In this case, number of people in the room, y = 20.6940 - 0.8462*x1 + 0.0002*x2 - 0.2323*x3 
# where x1 is Temperature, x2 is Humidity and x3 is Luminosity. Motion sensor is ignored.

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