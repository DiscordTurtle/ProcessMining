#Anything besides the model instance goes here, please work in classes and functions
import xmltodict
import json
import pm4py
import datetime
import pandas as pd


#everything that runs for the model goes here
class Model:
    
    
    # Convert string to datetime object
    def convert_to_datetime(self,date):
        try:
            date_object = datetime.datetime.strptime(date, '%Y-%m-%d %H:%M:%S.%f%z')
        except:
            date_object = datetime.datetime.strptime(date, '%Y-%m-%d %H:%M:%S%z')
        return date_object

    # Get time difference in seconds between two dates
    def get_time_difference_as_number(self,date1, date2):
        date1 = self.convert_to_datetime(date1)
        date2 = self.convert_to_datetime(date2)
        return (date2 - date1).total_seconds()

    # Export DataFrame to CSV
    def save_csv(DataFrame,file):
        DataFrame.to_csv(file, index=False)
        
    # Import CSV to DataFrame
    def get_csv(file): 
        return pd.read_csv(file, sep=',')
        
#everything that only runs once (like converting files) goes here
class Auxliary:

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
    def train_test_split(df, test_size=0.2):
        df = df.sample(frac=1).reset_index(drop=True)
        train_size = int(len(df) * (1 - test_size)) -1
        
        df = df.sort_values(by='case:REG_DATE')
        
        dftrain = df[:train_size]
        dftest = df[train_size:]

        print(df)
        #case attributes at which the split is made
        split_case_name = df.iloc[train_size]['case:concept:name']
        split_case_date = df.iloc[train_size]['case:REG_DATE']
        print(split_case_name)
        print(split_case_date)

        print(df.loc[df['case:concept:name'] == split_case_name, df['case:REG_DATE'] == split_case_date])

        #print(dftrain)
        #print(dftest)

        #sort the data by case:concept:name and date
        dftrainmax = dftrain.groupby(by='case:concept:name', as_index = False)['time:timestamp'].max()
        
        #defining split time variable to check for overlap and selecting only rows that don't overlap
        min_split_time = dftest['time:timestamp'].min()
        dftrain_no_overlap = dftrainmax[dftrainmax['time:timestamp']<min_split_time]


        dftrain_mask = dftrain['case:concept:name'].isin(list([dftrain_no_overlap['case:concept:name']]))



        #print(dftrainmax['time:timestamp']<min_split_time)
        #print(dftrain_mask)
        #print(dftrain)
        #print(list(dftrain_no_overlap['case:concept:name']))

        #Save the data
        #dftrain_no_overlap.to_csv('..\Model\Train_BPI_Challenge_2012.csv', index=False)
        #dftest.to_csv('..\Model\Test_BPI_Challenge_2012.csv', index=False)
        

