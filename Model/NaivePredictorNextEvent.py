#import libraries
import pandas as pd
import numpy as np

#import the neccessary module
from readCSV import importCSV
from OutputCSV import exportCSV

df = importCSV('BPI_Challenge_2012.csv')

#add a new column with the position of each case
df.at[0, 'position'] = 1
j = 2
for i in range(1, len(df.index)):
    if df.at[i, 'case:concept:name'] == df.at[i - 1, 'case:concept:name']:
        df.at[i, 'position'] = j
        j = j + 1
    else :
        df.at[i, 'position'] = 1
        j = 2

#the mode of the activities at a certain position
df_mode = df.groupby('position')['concept:name'].agg(pd.Series.mode)

#add a new column with the predicted activity of each case
for i in range(0, len(df.index)):
    if df.at[i, 'position'] + 1 <= len(df_mode.index):
        df.at[i, 'Next activity'] = df_mode.at[df.at[i, 'position'] + 1]

exportCSV(df)