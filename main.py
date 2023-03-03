#The model instance goes here
from helper import Model
from helper import Auxiliary


df = Model.get_csv(r'C:\Users\20212387\OneDrive - TU Eindhoven\Documents\Y2\DBL process mining\ProcessMining\Model\BPI_Challenge_2012.csv')
Auxiliary.train_test_split(df)


