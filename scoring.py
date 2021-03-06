from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
import pickle
import os
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import json



#################Load config.json and get path variables
with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path']) 
test_data_path = os.path.join(config['test_data_path'])
model_path = os.path.join(config['output_model_path'])


#################Function for model scoring
def score_model(mdl_path, data_to_score):
    #this function should take a trained model, load test data, and calculate an F1 score for the model relative to the test data
    #it should write the result to the latestscore.txt file

    if mdl_path is None: mdl_path = model_path
    with open(mdl_path + '/trainedmodel.pkl', 'rb') as file:
        model = pickle.load(file)
    
    if mdl_path is None:
        testdata = pd.read_csv(test_data_path + '/testdata.csv')
    else: 
        testdata = pd.read_csv(dataset_csv_path + '/finaldata.csv')

    X = testdata.loc[:,['lastmonth_activity','lastyear_activity','number_of_employees']]
    y = testdata['exited']

    predicted=model.predict(X)

    f1score=metrics.f1_score(predicted,y)
    

    with open(model_path+'/latestscore.txt','w') as f:
        f.write(str(f1score)+'\n')

    return str(f1score)


if __name__ == '__main__':
    score_model(None,None)
