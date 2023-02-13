# Method to split the csv into train and test data
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sb
import datetime 


# date1 = "2012-01-01 00:00:01.1133"

# date1_object = datetime.datetime.strptime(date1, r'%Y-%m-%d %H:%M:%S.%f')
# timestamp = date1_object.timestamp()
# print(timestamp)
csv_data = pd.read_csv('../2012-Data/BPI_Challenge_2012.csv')

# for entry in csv_data['time:timestamp']:
#     # csv_data['number_time:timestamp'] = csv_data.apply(lambda x: datetime.datetime.strptime(x, r'%Y-%m-%d %H:%M:%S.%f').timestamp())
#     csv_data[entry] = entry.split('+')[0]
for index, entry in csv_data['case:REG_DATE'].iteritems():
    # print(csv_data['time:timestamp'])
    csv_data['case:REG_DATE'].loc[index],_ = entry.split('+')
    # print(csv_data['time:timestamp'])
    try:
        csv_data['case:REG_DATE'].loc[index] = datetime.datetime.strptime(csv_data['case:REG_DATE'].loc[index], r'%Y-%m-%d %H:%M:%S.%f').timestamp()
    except:
        csv_data['case:REG_DATE'].loc[index] = datetime.datetime.strptime(csv_data['case:REG_DATE'].loc[index], r'%Y-%m-%d %H:%M:%S').timestamp()
        
    print(csv_data['case:REG_DATE'].at[index])

csv_data.to_csv('../2012-Data/Modified2_BPI_Challenge_2012.csv', index=False)
    
