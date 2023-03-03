import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

from readCSV import importCSV
from ConvertDatetime import convert_to_datetime


data = pd.DataFrame(importCSV("BPI_Challenge_2012.csv"))
data_pred = pd.DataFrame(importCSV("Tool_Prediction.csv"))
data_time = data.copy()
data_time['time:timestamp'] = data['time:timestamp'].map(convert_to_datetime)
data_time['Next time'] = data_pred[['Next time']]

print(data_pred.corr())

#print(data.describe(include='all'))
def format_event(x):
    if(x == "W_Nabellen incomplete dossiers"):
        return("W_NaID")
    if(x == "W_Nabellen offertes"):
        return("W_NaO")
    if(x == "O_SENT"):
        return("O_SB")
    return x[0:5:1]
data["concept:name:abreviation"] = data["concept:name"].map(format_event)
print(data.head())
print(data["case:concept:name"].nunique())
print(data["concept:name"].value_counts()["A_SUBMITTED"])
print(data["concept:name"].value_counts()["A_PARTLYSUBMITTED"])

print(data["concept:name"].unique())
print(data["concept:name"].describe())
print(data["concept:name:abreviation"].unique())
print(data["concept:name:abreviation"].describe())

print("set the data")

def plot_amount(data_column): 
    ax = sns.countplot(x=data_column)
    return ax
#plot_amount(data["concept:name:abreviation"])
#plt.show()
#Uncomment the plt.show() for the graph

y = data_pred['Next activity']
y_pred = data_pred['Predicted next activity']

def plot_confusion_matrix(y, y_pred):
    y = y.replace(np.nan, "NULL")
    y_pred = y_pred.replace(np.nan, 'NULL')
    cf_matrix = confusion_matrix(y, y_pred)
    ax = sns.heatmap(cf_matrix)
    return ax

def plot_confusion_matrix_percent(y, y_pred):
    y = y.replace(np.nan, "NULL")
    y_pred = y_pred.replace(np.nan, 'NULL')
    cf_matrix = confusion_matrix(y, y_pred)
    ax = sns.heatmap(cf_matrix/np.sum(cf_matrix), annot=True, fmt='.2%', cmap='Blues')
    return ax

def plot_pairplot(data, x_vars_v, y_vars_v, hue_v = None): 
    ax = sns.pairplot(data, x_vars = x_vars_v, y_vars = y_vars_v, hue = hue_v )
    return ax
#plot_pairplot(data_time, ['time:timestamp', 'concept:name', 'Next time'], ['time:timestamp', 'concept:name', 'Next time'])
#plt.show()
#print('pairplot incoming')
#plot_pairplot(data_time, ['time:timestamp','case:concept:name', 'Next time'], ['time:timestamp','case:concept:name', 'Next time'], 'concept:name')
#plt.show()

#plot_pairplot(data_time, ['Next time'], ['concept:name'])
#plt.show()
#print("done")
#print(data.corr())