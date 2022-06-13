
import pandas as pd
import numpy as np
import timeit
import os
import json
import pickle
import subprocess
import sys

##################Load config.json and get environment variables
with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path']) 
test_data_path = os.path.join(config['test_data_path'])
prod_deployment_path = os.path.join(config['prod_deployment_path'])

##################Function to get model predictions
def model_predictions(df_path):
    #read the deployed model and a test dataset, calculate predictions
    with open(prod_deployment_path+'/trainedmodel.pkl', 'rb') as file:
        model = pickle.load(file)

    if df_path is None: df_path = test_data_path+"/testdata.csv"
    df = pd.read_csv(df_path)
    predictions = model.predict(df.loc[:,['lastmonth_activity','lastyear_activity','number_of_employees']])

    return list(predictions), list(df['exited'].values)

##################Function to get summary statistics
def dataframe_summary():
    #calculate summary statistics here
    df = pd.read_csv(dataset_csv_path+'/finaldata.csv')
    df = df.drop(['exited'], axis=1)
    df = df.select_dtypes('number')
    stats_dict = {}
    for col in df.columns:
        mean = df[col].mean()
        median = df[col].median()
        std = df[col].std()

        stats_dict[col] = {'mean': mean, 'median': median, 'std': std}
    return stats_dict


##################Function to get missing data
def missing_data_stats():
    df = pd.read_csv(dataset_csv_path+'/finaldata.csv')
    missing_data_proportion = {}
    n_rows = df.shape[0]
    for col in df.columns:
        missing_data_proportion[col] = round(df[col].isna().sum()/n_rows*100, 2)

    return missing_data_proportion


##################Function to get timings
def execution_time():
    timing_list = []

    starttime = timeit.default_timer()
    os.system('python3 ingestion.py')
    timing_list.append(timeit.default_timer() - starttime)

    starttime = timeit.default_timer()
    os.system('python3 training.py')
    timing_list.append(timeit.default_timer() - starttime)
    
    return timing_list

##################Function to check dependencies
def outdated_packages_list():
    outdated_packages = subprocess.check_output(['pip', 'list', '--outdated']).decode(sys.stdout.encoding)
    
    return str(outdated_packages)


if __name__ == '__main__':
    model_predictions(None)
    dataframe_summary()
    missing_data_stats()
    execution_time()
    outdated_packages_list()
