#import libraries
import pandas as pd
import numpy as np

#import the neccessary module
from helper import Model
from helper import Auxiliary

# Modelling
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LinearRegression

#Evaluating
from sklearn.metrics import accuracy_score
from sklearn.metrics import r2_score

#Import CSV to DataFrame
df = Model.get_csv('BPI_Challenge_2012.csv')

unique_activities = df['concept:name'].unique()

#Create a copy of df_test before preprocessing data for visualization
df_train0, vizualization = Auxiliary.train_test_split(df)

#Preprocess data and then split it into train and test sets
df_train, df_test = Auxiliary.train_test_split(Auxiliary.preprocess_data(df))

#START of the implemenation of the Random Forest Classifier
print("START of the implemenation of the Random Forest Classifier")

#split the data into training and test sets
X_train_rf = df_train.drop(['org:resource', 'case:AMOUNT_REQ', 'Next Event', 'case:REG_DATE', 'time:timestamp'], axis=1)
X_test_rf = df_test.drop(['org:resource', 'case:AMOUNT_REQ', 'Next Event', 'case:REG_DATE', 'time:timestamp'], axis=1)
y_train_rf = df_train['Next Event']
y_test_rf = df_test['Next Event']

#create an instance of the Random Forest model and then fit this to our training data
rf = RandomForestClassifier()
rf.fit(X_train_rf, y_train_rf)

#predict the data
y_pred_rf = rf.predict(X_test_rf)

#compute the accuracy of the predictor
accuracy_rf = accuracy_score(y_test_rf, y_pred_rf)
print("Accuracy:", accuracy_rf)

vizualization['Next Event number'] = y_test_rf
vizualization['Predicted Event number'] = y_pred_rf

dictionary = {i : unique_activities[i] for i in range(unique_activities.size)}

vizualization['Next Event string'] = vizualization['Next Event number'].map(dictionary)
vizualization['Predicted Event string'] = vizualization['Predicted Event number'].map(dictionary)

#END of the implemenation of the Random Forest Classifier
print("END of the implemenation of the Random Forest Classifier")

#START of the implemenation of the Linear Regression
print("START of the implemenation of the Linear Regression")

# TO DO...

#END of the implemenation of the Linear Regression
print("END of the implemenation of the Linear Regression")

Model.save_csv(vizualization, 'final_df.csv')