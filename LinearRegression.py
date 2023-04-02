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


df = Model.get_csv('BPI_Challenge_2012.csv')
#df_train, df_test = Auxiliary.train_test_split(Auxiliary.preprocess_data(Model.get_csv('BPI_Challenge_2012.csv')))
#Preprocess data and then split it into train and test sets
df_train, df_test = Auxiliary.train_test_split(Auxiliary.preprocess_data(df))

print(df_train.columns)

#only select needed columns
df_train = df_train[['org:resource', 'case:concept:name', 'concept:name', 'month', 'day', 'Week Day', 'Next Time', 'lifecycle:transition', 
                        'A_SUBMITTED', 'A_PARTLYSUBMITTED', 'A_PREACCEPTED',
                        'W_Completeren aanvraag', 'A_ACCEPTED', 'O_SELECTED', 'A_FINALIZED',
                        'O_CREATED', 'O_SENT', 'W_Nabellen offertes', 'O_SENT_BACK',
                        'W_Valideren aanvraag', 'A_REGISTERED', 'A_APPROVED', 'O_ACCEPTED',
                        'A_ACTIVATED', 'O_CANCELLED', 'W_Wijzigen contractgegevens',
                        'A_DECLINED', 'A_CANCELLED', 'W_Afhandelen leads', 'O_DECLINED',
                        'W_Nabellen incomplete dossiers', 'W_Beoordelen fraude']]
df_test = df_test[['org:resource', 'case:concept:name', 'concept:name', 'month', 'day', 'Week Day', 'Next Time', 'lifecycle:transition', 
                        'A_SUBMITTED', 'A_PARTLYSUBMITTED', 'A_PREACCEPTED',
                        'W_Completeren aanvraag', 'A_ACCEPTED', 'O_SELECTED', 'A_FINALIZED',
                        'O_CREATED', 'O_SENT', 'W_Nabellen offertes', 'O_SENT_BACK',
                        'W_Valideren aanvraag', 'A_REGISTERED', 'A_APPROVED', 'O_ACCEPTED',
                        'A_ACTIVATED', 'O_CANCELLED', 'W_Wijzigen contractgegevens',
                        'A_DECLINED', 'A_CANCELLED', 'W_Afhandelen leads', 'O_DECLINED',
                        'W_Nabellen incomplete dossiers', 'W_Beoordelen fraude']]

#remove entries where there is NaN
df_train = df_train.dropna()
df_test = df_test.dropna()
df_train = df_train.replace(-1, 0)
df_test = df_test.replace(-1, 0)

#q__train_low = df_train["Next Time"].quantile(0.03)
q__train_hi  = df_train["Next Time"].quantile(0.97)

#df_train = df_train[(df_train["Next Time"] < q__train_hi) & (df_train["Next Time"] > q__train_low)]
df_train = df_train[df_train["Next Time"] < q__train_hi]
#q_test_low = df_test["Next Time"].quantile(0.01)
q_test_hi = df_test["Next Time"].quantile(0.97)
#df_test = df_test[(df_test["Next Time"] < q_test_hi) & (df_test["Next Time"] > q_test_low)]
df_test = df_test[df_test["Next Time"] < q_test_hi]

#split the data into training and test sets and drop some data
#x_train = df_train[['org:resource', 'lifecycle:transition','concept:name','case:AMOUNT_REQ','month', 'day']]
X_train = df_train[['org:resource', 'day', 'month', 'Week Day', 'lifecycle:transition', 
                        'A_SUBMITTED', 'A_PARTLYSUBMITTED', 'A_PREACCEPTED',
                        'W_Completeren aanvraag', 'A_ACCEPTED', 'O_SELECTED', 'A_FINALIZED',
                        'O_CREATED', 'O_SENT', 'W_Nabellen offertes', 'O_SENT_BACK',
                        'W_Valideren aanvraag', 'A_REGISTERED', 'A_APPROVED', 'O_ACCEPTED',
                        'A_ACTIVATED', 'O_CANCELLED', 'W_Wijzigen contractgegevens',
                        'A_DECLINED', 'A_CANCELLED', 'W_Afhandelen leads', 'O_DECLINED',
                        'W_Nabellen incomplete dossiers', 'W_Beoordelen fraude']]
y_train = df_train[['Next Time']]
#x_test = df_test[['org:resource', 'lifecycle:transition','concept:name','case:AMOUNT_REQ','month', 'day']]
X_test = df_test[['org:resource', 'day', 'month', 'Week Day', 'lifecycle:transition', 
                        'A_SUBMITTED', 'A_PARTLYSUBMITTED', 'A_PREACCEPTED',
                        'W_Completeren aanvraag', 'A_ACCEPTED', 'O_SELECTED', 'A_FINALIZED',
                        'O_CREATED', 'O_SENT', 'W_Nabellen offertes', 'O_SENT_BACK',
                        'W_Valideren aanvraag', 'A_REGISTERED', 'A_APPROVED', 'O_ACCEPTED',
                        'A_ACTIVATED', 'O_CANCELLED', 'W_Wijzigen contractgegevens',
                        'A_DECLINED', 'A_CANCELLED', 'W_Afhandelen leads', 'O_DECLINED',
                        'W_Nabellen incomplete dossiers', 'W_Beoordelen fraude']]
y_test = df_test[['Next Time']]

#normalize y valuesf

split_location = y_train.shape[0]
y_df = pd.concat([y_train, y_test])

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

scaler.fit(y_df)

y_df = scaler.transform(y_df)

#Resplit as numpy arrays
#y_train = y_df[0:split_location]
#y_test = y_df[split_location:]

X_train = X_train.values
X_test = X_test.values

temp_array = []
real_x_train = []
real_x_test = []
for index,value in enumerate(X_train):
    temp_array = []
    temp_array = value[0]
    temp_array = np.append(temp_array, value[1])
    temp_array = np.append(temp_array, value[2])
    real_x_train.append(temp_array)
for index,value in enumerate(X_test):
    temp_array = []
    temp_array = value[0]
    temp_array = np.append(temp_array, value[1])
    temp_array = np.append(temp_array, value[2])
    real_x_test.append(temp_array)

x_train = np.array(real_x_train)
x_test = np.array(real_x_test)


#create an instance of a Linear Regression model and then fit this to our training data
reg = LinearRegression()
reg.fit(X_train, y_train)

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

compare_result.to_csv('Linear_Regression_Prediction_Ground_Truth.csv', index = False)
