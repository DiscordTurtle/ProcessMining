import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from readCSV import importCSV


data = pd.DataFrame(importCSV("BPI_Challenge_2012.csv"))
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
sns.countplot(x=data["concept:name"])
#plt.show()
sns.countplot(x=data["concept:name:abreviation"])
#plt.show()
#Uncomment the plt.show() for the graph

