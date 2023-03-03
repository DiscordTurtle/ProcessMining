#import libraries
import pandas as pd
import numpy as np

#import the neccessary module
from helper import Model
from helper import Auxiliary

# Modelling
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

#read the data set
df_train = Auxiliary.preprocess_data(Model.get_csv('BPI_Challenge_2012_train.csv'))
df_test = Auxiliary.preprocess_data(Model.get_csv('BPI_Challenge_2012_test.csv'))

#split the data into training and test sets
X_train = df_train.drop(['org:resource', 'case:AMOUNT_REQ', 'Next Event', 'case:REG_DATE'], axis=1)
X_test = df_test.drop(['org:resource', 'case:AMOUNT_REQ', 'Next Event', 'case:REG_DATE'], axis=1)
y_train = df_train['Next Event']
y_test = df_test['Next Event']

#create an instance of the Random Forest model and then fit this to our training data
rf = RandomForestClassifier()
rf.fit(X_train, y_train)

#predict the data
y_pred = rf.predict(X_test)

#compute the accuracy of the predictor
accuracy = accuracy_score(y_test, y_pred)
print(y_pred.size)
print("Accuracy:", accuracy)

# X_test['Next Event'] = y_test
# X_test['Predicted Event'] = y_pred

# exportCSV(X_test)