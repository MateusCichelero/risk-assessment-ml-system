import pandas as pd
import numpy as np
import os
import json
from datetime import datetime




#############Load config.json and get input and output paths
with open('config.json','r') as f:
    config = json.load(f) 

input_folder_path = config['input_folder_path']
output_folder_path = config['output_folder_path']



#############Function for data ingestion
def merge_multiple_dataframe():
    #check for datasets, compile them together, and write to an output file
    filenames = os.listdir(input_folder_path)
    df_list = pd.DataFrame()
    for each_filename in filenames:
        df1 = pd.read_csv(input_folder_path+'/'+each_filename)
        df_list=df_list.append(df1)
    

    result=df_list.drop_duplicates()
    result.to_csv(output_folder_path+'/finaldata.csv', index=False)

    MyFile=open(output_folder_path+'/ingestedfiles.txt','w')
    for element in filenames:
        MyFile.write(str(element)+'\n')



if __name__ == '__main__':
    merge_multiple_dataframe()
