#import libraries
import pandas as pd
import numpy as np

#import the neccessary module
from helper import Model
from helper import Auxiliary

# Modelling
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt

#HyperParameterTuning
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import GridSearchCV

df_train, df_test = Auxiliary.train_test_split(Auxiliary.preprocess_data(Model.get_csv('BPI_Challenge_2012.csv')))
#read the data set
# df_train = Auxiliary.preprocess_data(Model.get_csv('BPI_Challenge_2012_train.csv'))
# df_test = Auxiliary.preprocess_data(Model.get_csv('BPI_Challenge_2012_test.csv'))

#split the data into training and test sets
X_train = df_train.drop(['org:resource', 'case:AMOUNT_REQ', 'Next Event', 'case:REG_DATE', 'time:timestamp'], axis=1)
X_test = df_test.drop(['org:resource', 'case:AMOUNT_REQ', 'Next Event', 'case:REG_DATE', 'time:timestamp'], axis=1)
y_train = df_train['Next Event']
y_test = df_test['Next Event']

#Scaling the data
scaler = StandardScaler()
#X_train = scaler.fit_transform(X_train)
#X_test = scaler.transform(X_train)
#y_train = scaler.transform(X_train)
#y_test = scaler.transform(X_train)


print(X_train.columns)

#create an instance of the Random Forest model and then fit this to our training data
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

#predict the data
y_pred = knn.predict(X_test)

#compute the accuracy of the predictor
accuracy = accuracy_score(y_test, y_pred)
print(y_pred.size)
print("Accuracy:", accuracy)

# X_test['Next Event'] = y_test
# X_test['Predicted Event'] = y_pred

# exportCSV(X_test)



# calculating the accuracy of models with different values of k
mean_acc = np.zeros(30)
for i in range(1,31):
    #Train Model and Predict  
    knn = KNeighborsClassifier(n_neighbors = i).fit(X_train,y_train)
    yhat= knn.predict(X_test)
    mean_acc[i-1] = metrics.accuracy_score(y_test, yhat)

print(mean_acc)
loc = np.arange(1,31,step=1.0)
plt.figure(figsize = (10, 6))
plt.plot(range(1,31), mean_acc)
plt.xticks(loc)
plt.xlabel('Number of Neighbors ')
plt.ylabel('Accuracy')
plt.show()

#List Hyperparameters that we want to tune.
leaf_size = list(range(1,50))
n_neighbors = list(range(1,30))
p=[1,2]
metric = ['euclidean', 'manhattan', 'minkowski']
weights = ['uniform', 'distance']

#Convert to dictionary
grid = dict(leaf_size=leaf_size, n_neighbors=n_neighbors, p=p, metric = metric, weights = weights)

#Create new KNN object
knn_2 = KNeighborsClassifier()

#Do the grid search
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
grid_search = GridSearchCV(estimator=knn_2, param_grid=grid, n_jobs=-1, cv=cv, scoring='accuracy',error_score=0)
grid_result = grid_search.fit(X_train, y_train)

# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))

