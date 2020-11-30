#Import the Libraries

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

#Importing the Dataset

dataset = pd.read_csv('/Users/nicomellein/Desktop/Machine Learning A-Z (Codes and Datasets)/Part 2 - Regression/Selection Model /Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Splitting the dataset into the Training set and Test set

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


# Training the Decision Tree Regression model on the Training set

from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state = 0)
regressor.fit(X_train, y_train)

# Predicting the Test set results

y_pred = regressor.predict(X_test)
np.set_printoptions(precision=2)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))

# Evaluating the Model Performance

from sklearn.metrics import r2_score
print(r2_score(y_test, y_pred))