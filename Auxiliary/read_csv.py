# Method to read a CSV file and make a corresponding Pandas DataFrame
import pandas as pd

def importCSV(file): 
    return pd.read_csv(file, sep=',')

print(importCSV('BPI_Challenge_2012.csv'))
