import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
 
#Data Preparation
df = pd.read_csv("C:/Users/HP/Documents/bank.csv")

X = df[df.columns.drop('y')]
X = pd.get_dummies(X)
y = df['y']

#test for multicolinearity 

print(df.corr()>abs(.5))

#Use 70% of dataset as training set and remaining 30% as testing set
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.30,random_state=0)

#model fitting 

logistic_regression= LogisticRegression(max_iter=100000)
logistic_regression.fit(X_train,y_train)

#test model on unseen data
y_pred=logistic_regression.predict(X_test)

#model evaluation

confusion_matrix = pd.crosstab(y_test, y_pred, rownames=['Actual'], colnames=['Predicted'])
print(confusion_matrix)
print('Accuracy: ',metrics.accuracy_score(y_test, y_pred))
