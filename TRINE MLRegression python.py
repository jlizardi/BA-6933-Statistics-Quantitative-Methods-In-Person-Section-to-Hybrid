import pandas as pd
import numpy as np


#Data Preparation
dataset = pd.read_csv('C:/Users/HP/Documents/50_Startups.csv')
dataset.describe()
X = dataset[dataset.columns.drop('Profit')]
X = pd.get_dummies(X)
y = dataset['Profit']

#Use 70% of dataset as training set and remaining 30% as testing set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
from sklearn.linear_model import LinearRegression

#test for multicolinearity 
print(dataset.corr()>abs(.5))
print(dataset.corr())

print(X.corr()>abs(.5))
print(X.corr())


import seaborn as sns
import matplotlib.pyplot as plt
f,ax=plt.subplots(figsize=(20,20))
sns.heatmap(X.corr(),annot=True)

import seaborn as sns
import matplotlib.pyplot as plt
f,ax=plt.subplots(figsize=(20,20))
sns.heatmap(X.corr()>abs(.5),annot=True)

import seaborn as sns
import matplotlib.pyplot as plt
f,ax=plt.subplots(figsize=(20,20))
sns.heatmap(dataset.corr(),annot=True)

import seaborn as sns
import matplotlib.pyplot as plt
f,ax=plt.subplots(figsize=(20,20))
sns.heatmap(dataset.corr()>abs(.5),annot=True)

regressor = LinearRegression()
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)

df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
print(df)

from sklearn import metrics
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
print('R^2:', metrics.r2_score(y_test, y_pred))
