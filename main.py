#The model instance goes here
from helper import Model
from helper import Auxiliary
from helper import Graphs


df_train = Model.get_csv(r'C:\Users\20212387\OneDrive - TU Eindhoven\Documents\Y2\DBL process mining\ProcessMining\BPI_Challenge_2012_train.csv')
df_test = Model.get_csv(r'C:\Users\20212387\OneDrive - TU Eindhoven\Documents\Y2\DBL process mining\ProcessMining\BPI_Challenge_2012_test.csv')
#Auxiliary.train_test_split(df)

Graphs.tt_split_graph(df_train, df_test)

