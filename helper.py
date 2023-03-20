#Anything besides the model instance goes here, please work in classes and functions
import xmltodict
import json
import pm4py
#import matplotlib as plt
import datetime
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix

#everything that runs for the model goes here
class Model:
    
    
    # Convert string to datetime object
    def convert_to_datetime(date):
        try:
            date_object = datetime.datetime.strptime(date, '%Y-%m-%d %H:%M:%S.%f%z')
        except:
            date_object = datetime.datetime.strptime(date, '%Y-%m-%d %H:%M:%S%z')
        return date_object

    # Get time difference in seconds between two dates
    def get_time_difference_as_number(date1, date2):
        date1 = Model.convert_to_datetime(date1)
        date2 = Model.convert_to_datetime(date2)
        return (date2 - date1).total_seconds()
    
    def convert_to_seconds(date):
        return Model.get_time_difference_as_number('1970-01-01 00:00:00.000000+00:00', date)
    
    def get_year(date):
        return Model.convert_to_datetime(date).strftime("%Y")
    
    def get_month(date):
        return Model.convert_to_datetime(date).strftime("%m")
    
    def get_day(date):
        return Model.convert_to_datetime(date).strftime("%d")
    
    def get_hour(date):
        return Model.convert_to_datetime(date).strftime("%H")
    
    def get_minute(date):
        return Model.convert_to_datetime(date).strftime("%M")
    
    def get_second(date):
        return Model.convert_to_datetime(date).strftime("%S")

    # Export DataFrame to CSV
    def save_csv(DataFrame,file_path):
        DataFrame.to_csv(file_path, index=False, sep=',')
        
    # Import CSV to DataFrame
    def get_csv(file_path): 
        return pd.read_csv(file_path, sep=',')
    
    
    def naive_predictor(df):
        #put naive predictor here
        pass
        
#everything that only runs once (like converting files) goes here
class Auxiliary:

    # Convert XML to JSON
    def XML_to_JSON(xml_file_path):
        
        with open(xml_file_path) as xml_file:
            data_dict = xmltodict.parse(xml_file.read())
            json_data = json.dumps(data_dict)
            
            with open(xml_file_path[:-4] + '.json', "w") as json_file:
                json_file.write(json_data)

    # Convert XES to CSV
    def XES_to_CSV(xes_file):
        log = pm4py.read_xes(xes_file)
        pd = pm4py.convert_to_dataframe(log)
        pd.to_csv(xes_file[:-4] + '.csv', index=False)

    # split the data into training and test data
    def train_test_split(df, test_size=0.20):
        print("Start splitting data")
        train_size = int(len(df) * (1 - test_size)) -1
    
        #print(train_size)

        df_train = df[:train_size]
        df_test = df[train_size:]

        #case attributes at which the split is made
        split_case_name = df.iloc[train_size]['case:concept:name']
        split_case_date = df.iloc[train_size]['case:REG_DATE']

        #index of first occurence of entry on which to split
        split_index = df.index[(df['case:concept:name'] == split_case_name) & (df['case:REG_DATE'] == split_case_date)].min()

        df_train = df[0:split_index]
        df_test = df[split_index:len(df)]

        # Model.save_csv(df_train, 'BPI_Challenge_2012_train.csv')
        # print(df_train)

        min_split_time = df_test['time:timestamp'].min()

        
        
        

        #get the max timestamp for each case in the training data
        df_train_max = df_train.groupby(by='case:concept:name', as_index = False)['time:timestamp'].max()
        
        
        #defining split time variable to check for overlap and selecting only rows that don't overlap
        min_split_time = df_test['time:timestamp'].min()
        df_train_no_overlap = df_train_max[df_train_max['time:timestamp']<min_split_time]


        df_train_mask = df_train['case:concept:name'].isin(list(df_train_no_overlap['case:concept:name']))

        df_train = df_train[df_train_mask]

        #remove last percentage of train data to compensate for smaller traces (positive range so far 5-8%)
        percentage_removed = 5
        train_size_final = int(len(df_train) * (1 - (percentage_removed/100))) -1
        #print(df_train)
        df_train = df_train[:train_size_final]
        #print(df_train)
        #removing last case so df only contains full cases
        last_case = df_train.iloc[len(df_train)-1]['case:concept:name']
        df_train = df_train[df_train['case:concept:name'] != last_case]


        # print(df_train)
        # print(df_test)

        #Save the data
        # Model.save_csv(df_train, 'BPI_Challenge_2012_train.csv')
        # Model.save_csv(df_test, 'BPI_Challenge_2012_test.csv')
        print("Finished splitting data")
        return df_train, df_test

    def preprocess_data(df):
        print("Start preprocessing data")
        lengthOfDf = len(df.index)

        #map the lifecycle:transition
        df['lifecycle:transition'] = df['lifecycle:transition'].map({'COMPLETE':2,'SCHEDULE':0,'START':1})

        #create a dictionary for the concept:name
        unique_activities = df['concept:name'].unique()
        # create a new column for each event
        # for i in unique_activities:
        #     df[i] = [1 if x == i else 0 for x in df['concept:name']]
        dictionary = {unique_activities[i]: i for i in range(unique_activities.size)}
        #map the concept:name
        df['concept:name'] = df['concept:name'].map(dictionary)

        #create new columns for the date
        df['year'] = [Model.get_year(x) for x in df['time:timestamp']]
        df['month'] = [Model.get_month(x) for x in df['time:timestamp']]
        df['day'] = [Model.get_day(x) for x in df['time:timestamp']]
        df['hour'] = [Model.get_hour(x) for x in df['time:timestamp']]
        df['minute'] = [Model.get_minute(x) for x in df['time:timestamp']]
        df['second'] = [Model.get_second(x) for x in df['time:timestamp']]

        for i in range(lengthOfDf):
            #create new column for the ground truth
            if i < lengthOfDf - 1:
                if df.at[i, 'case:concept:name'] == df.at[i + 1, 'case:concept:name']:
                    df.at[i, 'Next Event'] = df.at[i + 1, 'concept:name']
                else:
                    df.at[i, 'Next Event'] = -1
            else:
                df.at[i, 'Next Event'] = -1
        print("Finished preprocessing data")

        df['time:timestamp'] = [Model.convert_to_datetime(x) for x in df['time:timestamp']]
        df['case:REG_DATE'] = [Model.convert_to_datetime(x) for x in df['case:REG_DATE']]
        df['time to complete'] = (df['time:timestamp'] - df['case:REG_DATE']).dt.seconds

        return df
    
    
    
    
class Graphs:

    def tt_split_graph(df, df_tr, df_te):
        
        #defining dataframes
        df_removed = df
        df_train = df_tr
        df_test = df_te

        #dataframe to mark removed entries for tt split
        df_mask = df_removed['case:concept:name'].isin(df_train['case:concept:name'])
        df_removed = df_removed[df_mask]

        #selecting a workable amount of data
        df_train_sample = df_tr[df_tr.index % 500 == 0]
        df_test_sample =  df_te[df_te.index % 500 == 0]

        #selecting only the case identifyers of this data
        df_train_case = df_train_sample['case:concept:name']
        df_test_case = df_test_sample['case:concept:name']

        #selecting every occurence of the selected cases
        df_train = df_train[df_train['case:concept:name'].isin(list(df_train_case))]
        df_test = df_test[df_test['case:concept:name'].isin(list(df_test_case))]

        #adding column for only y/m/d of cases
        df_train['time:y/m/d'] = df_train['time:timestamp'].str[:10]
        df_test['time:y/m/d'] = df_test['time:timestamp'].str[:10]

        #plotting the 2 datasets into 1 figure
        fig, ax1 = plt.subplots(nrows=1, ncols=1, sharex=True, sharey=True, figsize = (35,10))
        #fixes the x label
        ax1.set_xticklabels(df_train['time:y/m/d'])
        ax1.xaxis.set_major_locator(plt.MaxNLocator(10))

        #df_train.plot(kind='scatter', x='time:timestamp', y='case:concept:name', s=2, ax = ax1, figsize = (25, 10))
        #df_test.plot(kind='scatter', x='time:timestamp', y='case:concept:name', s=2, ax = ax1, c='red')

        sns.scatterplot(data=df_train, x ='time:timestamp', y ='case:concept:name', s=40, ax = ax1, hue ='concept:name', linewidth=0)
        sns.scatterplot(data=df_test, x ='time:timestamp', y ='case:concept:name', s=40, ax = ax1, hue = 'concept:name', linewidth=0)

        ax1.get_legend().remove()
        #adding lines to define tt split
        ax1.axvline(df_test['time:timestamp'].min())
        ax1.axhline(df_test['case:concept:name'].min())
        
        #drawing lines for removed cases
        #for case in list(df_removed['case:concept:name']):
        #    ax1.axhline(case, color = 'red')
        
        plt.show()

        # data = pd.DataFrame(Model.get_csv("BPI_Challenge_2012.csv"))
        # data_pred = pd.DataFrame(Model.get_csv("Tool_Prediction.csv"))
        # data_time = data.copy()
        # data_time['time:timestamp'] = data['time:timestamp'].map(Model.convert_to_datetime)
        # data_time['Next time'] = data_pred[['Next time']]

        # print(data_pred.corr())

    #print(data.describe(include='all'))
    def format_event(x):
        if(x == "W_Nabellen incomplete dossiers"):
            return("W_NaID")
        if(x == "W_Nabellen offertes"):
            return("W_NaO")
        if(x == "O_SENT"):
            return("O_SB")
        return x[0:5:1]
        data["concept:name:abreviation"] = data["concept:name"].map(format_event)

    print("set the data")

    def plot_amount(data_column): 
        ax = sns.countplot(x=data_column)
        return ax
    #plot_amount(data["concept:name:abreviation"])
    #plt2.show()
    #Uncomment the plt.show() and above for the graph

        y = data_pred['Next activity']
        y_pred = data_pred['Predicted next activity']

    def plot_confusion_matrix(y, y_pred):
        y = y.replace(np.nan, "NULL")
        y_pred = y_pred.replace(np.nan, 'NULL')
        cf_matrix = confusion_matrix(y, y_pred)
        ax = sns.heatmap(cf_matrix)
        return ax

    def plot_confusion_matrix_percent(y, y_pred):
        y = y.replace(np.nan, "NULL")
        y_pred = y_pred.replace(np.nan, 'NULL')
        cf_matrix = confusion_matrix(y, y_pred)
        ax = sns.heatmap(cf_matrix/np.sum(cf_matrix), annot=True, fmt='.2%', cmap='Blues')
        return ax

    def plot_pairplot(data, x_vars_v, y_vars_v, hue_v = None): 
        ax = sns.pairplot(data, x_vars = x_vars_v, y_vars = y_vars_v, hue = hue_v )
        return ax
    #plot_pairplot(data_time, ['time:timestamp', 'concept:name', 'Next time'], ['time:timestamp', 'concept:name', 'Next time'])
    #plt2.show()
    #print('pairplot incoming')
    #plot_pairplot(data_time, ['time:timestamp','case:concept:name', 'Next time'], ['time:timestamp','case:concept:name', 'Next time'], 'concept:name')
    #plt2.show()

    #plot_pairplot(data_time, ['Next time'], ['concept:name'])
    #plt2.show()
    #print("done")
    #print(data.corr())
        
def main():
    pass


if __name__ == "__main__":
    main()