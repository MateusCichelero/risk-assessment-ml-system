from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
import pickle
import diagnostics 
import json
import os
import scoring



######################Set up variables for use in our script
app = Flask(__name__)

with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path']) 

prediction_model = None


#######################Prediction Endpoint
@app.route("/prediction", methods=['POST','OPTIONS'])
def predict():        
    dataset_path = request.json.get('dataset_path')
    y_pred, _ = diagnostics.model_predictions(dataset_path)
    return str(list(y_pred))

#######################Scoring Endpoint
@app.route("/scoring", methods=['GET','OPTIONS'])
def score():        
    return str(scoring.score_model())

#######################Summary Statistics Endpoint
@app.route("/summarystats", methods=['GET','OPTIONS'])
def summarystats():        
    summary_stats = diagnostics.dataframe_summary()
    return summary_stats

#######################Diagnostics Endpoint
@app.route("/diagnostics", methods=['GET','OPTIONS'])
def diag_metrics():        
    diagnostics_dict = {}
    diagnostics_dict['execution_time'] = diagnostics.execution_time()
    diagnostics_dict['missing_data_stats'] = diagnostics.missing_data_stats()
    diagnostics_dict['outdated_packages_list'] = diagnostics.outdated_packages_list()

    return diagnostics_dict

if __name__ == "__main__":    
    app.run(host='0.0.0.0', port=8000, debug=True, threaded=True)
