import pandas as pd

def exportCSV(DataFrame):
    DataFrame.to_csv("Tool_Prediction.csv", index=False)

# data = { "Event" : ['A','B','C'], "Time" : [2, 10, 7]}
# df = pd.DataFrame(data)
# exportCSV(df)
# Uncomment the lines above to export a dummy array.