import training
import scoring
import deployment
import diagnostics
import reporting
import ingestion
import json
import os

#############Load config.json and get input and output paths
with open('config.json','r') as f:
    config = json.load(f) 

prod_deployment_path = config['prod_deployment_path']
input_folder_path = config['input_folder_path']
output_folder_path = config['output_folder_path']



def main():
    # read currently used dataset files
    with open(os.path.join(prod_deployment_path, "ingestedfiles.txt")) as file:
        current_files = {line.strip('\n') for line in file.readlines()[1:]}
    
    # list new files
    new_files = set(os.listdir(input_folder_path))

    # checks if new files are different from already used ones, 
    # if they are the same, stop the process
    if len(new_files.difference(current_files)) == 0:
        return None

##################Checking for model drift
#check whether the score from the deployed model is different from the score from the model that uses the newest ingested data
    with open(prod_deployment_path + "/latestscore.txt", "r") as f:
        current_f1 = float(f.read())

    ingestion.merge_multiple_dataframe()
    new_f1 = float(scoring.score_model(prod_deployment_path, output_folder_path))
    
    print(f'oldf1: {current_f1}, newf1:{new_f1}')
##################Deciding whether to proceed, part 2
#if you found model drift, you should proceed. otherwise, do end the process here
    if new_f1 >= current_f1:
        return None

    training.train_model()

##################Re-deployment
#if you found evidence for model drift, re-run the deployment.py script
    deployment.deploy_files()

##################Diagnostics and reporting
#run diagnostics.py and reporting.py for the re-deployed model
    reporting.score_model()
    os.system("python3 apicalls.py")


if __name__ == '__main__':
    main()



