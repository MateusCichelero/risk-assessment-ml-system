import requests
import os
import json

with open('config.json','r') as f:
    config = json.load(f) 

model_path = os.path.join(config['output_model_path'])



#Specify a URL that resolves to your workspace
URL = "http://127.0.0.1:8000"

headers = {'Content-type': 'application/json', 'Accept': 'text/plain'}

#Call each API endpoint and store the responses
response1 = requests.post(f'{URL}/prediction',json={"dataset_path": "testdata/testdata.csv"}, headers=headers).text
response2 = requests.get(f'{URL}/scoring', headers=headers).text
response3 = requests.get(f'{URL}/summarystats', headers=headers).text
response4 = requests.get(f'{URL}/diagnostics', headers=headers).text

#combine all API responses
responses = f'{response1}\n\n\n{response2}\n\n\n{response3}\n\n\n{response4}'

#write the responses to your workspace
with open(model_path+'/apireturns.txt','w') as f:
    f.write(responses)
