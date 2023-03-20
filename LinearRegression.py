#import libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

#import the neccessary module
from helper import Model
from helper import Auxiliary

# Modelling
from sklearn.linear_model import LinearRegression

#Evaluating
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import r2_score

#df_train, df_test = Auxiliary.train_test_split(Auxiliary.preprocess_data(Model.get_csv('BPI_Challenge_2012.csv')))
#read the data set
df_train = Auxiliary.preprocess_data(Model.get_csv('BPI_Challenge_2012_train.csv'))
df_test = Auxiliary.preprocess_data(Model.get_csv('BPI_Challenge_2012_test.csv'))




# encode the concept:name as integer values
df_encoded_train = df_train.copy()

lengthOfDf = len(df_encoded_train.index)

#add a new column with the position of each case
df_encoded_train.at[0, 'position'] = 1
j = 2
for i in range(1, lengthOfDf):
    if df_encoded_train.at[i, 'case:concept:name'] == df_encoded_train.at[i - 1, 'case:concept:name']:
        df_encoded_train.at[i, 'position'] = j
        j = j + 1
    else :
        df_encoded_train.at[i, 'position'] = 1
        j = 2


#the mode of the activities at a certain position
df_mode = df_encoded_train.groupby('position')['concept:name'].agg(pd.Series.mode)

#add a new column with the predicted time of each case and the ground truth
for i in range(lengthOfDf - 1):
    if df_encoded_train.at[i, 'position'] + 1 <= len(df_mode.index):
        if (df_encoded_train.at[i + 1, 'position']) == 1:
            df_encoded_train.at[i, 'Next time'] = 0
        else:
            df_encoded_train.at[i, 'Next time'] = Model.get_time_difference_as_number(df_encoded_train.at[i, 'time:timestamp'], df_encoded_train.at[i + 1, 'time:timestamp'])

#same as above but now for testing data

# encode the concept:name as integer values
df_encoded_test = df_test.copy()
lengthOfDf = len(df_encoded_test.index)

#add a new column with the position of each case
df_encoded_test.at[0, 'position'] = 1
j = 2
for i in range(1, lengthOfDf):
    if df_encoded_test.at[i, 'case:concept:name'] == df_encoded_test.at[i - 1, 'case:concept:name']:
        df_encoded_test.at[i, 'position'] = j
        j = j + 1
    else :
        df_encoded_test.at[i, 'position'] = 1
        j = 2


#the mode of the activities at a certain position
df_mode = df_encoded_test.groupby('position')['concept:name'].agg(pd.Series.mode)

#add a new column with the predicted time of each case and the ground truth
for i in range(lengthOfDf - 1):
    if df_encoded_test.at[i, 'position'] + 1 <= len(df_mode.index):
        if (df_encoded_test.at[i + 1, 'position']) == 1:
            df_encoded_test.at[i, 'Next time'] = 0
        else:
            df_encoded_test.at[i, 'Next time'] = Model.get_time_difference_as_number(df_encoded_test.at[i, 'time:timestamp'], df_encoded_test.at[i + 1, 'time:timestamp'])


data_to_drop = ['org:resource', 'lifecycle:transition', 'case:AMOUNT_REQ', 'Next Event', 'case:REG_DATE', 'concept:name', 'case:concept:name', 'time:timestamp', 'position', 'minute',
       'second','year', 'month', 'hour']

#split the data into training and test sets and drop some data
X_train = df_encoded_train.drop(data_to_drop, axis=1)
X_train = X_train.drop(['W_Wijzigen contractgegevens'], axis = 1)
X_test = df_encoded_test.drop(data_to_drop, axis=1)
X_train = X_train.drop(['Next time'], axis=1)
X_test = X_test.drop(['Next time'], axis=1)

y_train = df_encoded_train['Next time']
y_test = df_encoded_test['Next time']

#fill all empty values with nan
X_train = X_train.replace(r'^\s*$', np.nan, regex=True)
X_test = X_test.replace(r'^\s*$', np.nan, regex=True)
y_train = y_train.replace(r'^\s*$', np.nan, regex=True)
y_test = y_test.replace(r'^\s*$', np.nan, regex=True)


#Fill all nan values with 0
X_train = X_train.replace(np.nan,0)
X_test = X_test.replace(np.nan,0)
y_train = y_train.replace(np.nan,0)
y_test = y_test.replace(np.nan,0)

print(X_train.isnull().values.any())
print(y_train.isnull().values.any())

print(X_train.columns)
print(X_test.columns)

#create an instance of a Linear Regression model and then fit this to our training data
reg = LinearRegression()
reg.fit(X_train, y_train)

#Put X_test to have the feature names to be in the same order
X_test = X_test[['A_SUBMITTED', 'A_PARTLYSUBMITTED', 'A_PREACCEPTED',
       'W_Completeren aanvraag', 'A_ACCEPTED', 'O_SELECTED', 'A_FINALIZED',
       'O_CREATED', 'O_SENT', 'W_Nabellen offertes', 'O_SENT_BACK',
       'W_Valideren aanvraag', 'A_REGISTERED', 'A_APPROVED', 'O_ACCEPTED',
       'A_ACTIVATED', 'O_CANCELLED', 'A_DECLINED', 'A_CANCELLED',
       'W_Afhandelen leads', 'O_DECLINED', 'W_Nabellen incomplete dossiers',
       'W_Beoordelen fraude','day']]
#predict the data
y_pred_train = reg.predict(X_train)
y_pred = reg.predict(X_test)

#compute the accuracy of the predictor
r2score_train = r2_score(y_train, y_pred_train)
r2score = r2_score(y_test, y_pred)
print(y_pred.size)
print("\nR2-Score:",  r2score, "\nR2-score train:", r2score_train)

# Export predicted data vs Ground truth
compare_result = pd.DataFrame()
compare_result['Next time'] = y_test
compare_result['Predicted Event'] = y_pred
X_train['concept:name'] = df_encoded_train[['concept:name']]
X_test['concept:name'] = df_encoded_test[['concept:name']]
X_test['Predicted Event'] = y_pred
X_test['Next time'] = y_test 

compare_result.to_csv('Linear_Regression_Prediction_Ground_Truth.csv', index = False)

plt.scatter(X_train['concept:name'], y_train,color='g') 
plt.plot(X_test['concept:name'], y_pred,color='k') 
plt.show()