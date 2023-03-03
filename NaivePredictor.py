#import libraries
import pandas as pd
import numpy as np

#############PLEASE PUT THIS IN EITHER HELPER OR MAIN#############
# (and make a method for it)


#import the neccessary module
from helper import Model, Auxliary

df = Model.get_csv('BPI_Challenge_2012.csv')
lengthOfDf = len(df.index)

#add a new column with the position of each case
df.at[0, 'position'] = 1
j = 2
for i in range(1, lengthOfDf):
    if df.at[i, 'case:concept:name'] == df.at[i - 1, 'case:concept:name']:
        df.at[i, 'position'] = j
        j = j + 1
    else :
        df.at[i, 'position'] = 1
        j = 2

#the mode of the activities at a certain position
df_mode = df.groupby('position')['concept:name'].agg(pd.Series.mode)

for i in range(1, len(df_mode.index)):
    if df_mode.map(type).iloc[i] != str:
        df_mode.iloc[i] = df_mode.iloc[i][0]

lengthOfUniquePosition = len(df['position'].unique())

nextTime = [0 for i in range(lengthOfUniquePosition)]
nextTimeCount = [0 for i in range(len(nextTime))]

for i in range(lengthOfDf):
    position = df.at[i, 'position'].astype(int)
    if position != 1:
        nextTime[position - 1] += Model.get_time_difference_as_number(df.at[i - 1, 'time:timestamp'], df.at[i, 'time:timestamp'])
        nextTimeCount[position - 1] += 1

for i in range(1, lengthOfUniquePosition):
    nextTime[i] = nextTime[i] / nextTimeCount[i]

#add a new column with the predicted activity of each case and the ground truth
for i in range(lengthOfDf - 1):
    if df.at[i, 'position'] + 1 <= len(df_mode.index):
        df.at[i, 'Next activity'] = df.at[i + 1, 'concept:name']
        df.at[i, 'Predicted next activity'] = df_mode.at[df.at[i, 'position'] + 1]
        df.at[i, 'Next time'] = Model.get_time_difference_as_number(df.at[i, 'time:timestamp'], df.at[i + 1, 'time:timestamp'])
        df.at[i, 'Predicted next time'] = nextTime[df.at[i, 'position'].astype(int)]


Model.save_csv(df, 'BPI_Challenge_2012.csv')