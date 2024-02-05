import json
import re
import os
import sys
import logging
import shutil
import pandas as pd
from sklearn import metrics
import subprocess
import training
import scoring
import deployment
import diagnostics
import reporting
import ingestion

##################Check and read new data
#first, read ingestedfiles.txt

#second, determine whether the source data folder has files that aren't listed in ingestedfiles.txt
def check_new_data(input_folder_path, prod_deployment_path):
    # Check and read new data
    filepath = os.path.join(prod_deployment_path, 'ingestedfiles.txt')

    ingestedfiles = []
    
    if os.path.exists(filepath):
        with open(filepath, 'r') as f:
            lines = f.readlines()
            for line in lines:
                match = re.search(r'((.+)\.csv)', line)
                if match:
                    ingestedfiles.append(match.group(1).strip())

    new_files = []

    for file in os.listdir(input_folder_path):
        match = re.search(r'((.+)\.csv)', file)
        if match.group(1).strip() not in ingestedfiles:
            new_files.append(file)
            ingestedfiles.append(match.group(1).strip())

    return new_files


def ingest_new_data(files, input_folder_path, output_folder_path):
    # Ingest new data
    if files:
        logging.info("ingesting new files")
        ingestion.merge_multiple_dataframe(input_folder_path, output_folder_path)
        return True
    else:
        logging.info("No new files")
        return False


def check_model_drift(move_to_next_step, output_folder_path, prod_deployment_path):
    latest_score_file = os.path.join(prod_deployment_path, 'latestscore.txt')
    final_data_file = os.path.join(output_folder_path, 'finaldata.csv')
    
    with open(latest_score_file, 'r') as f:
        latest_score = float(f.read().strip())

    dataset = pd.read_csv(final_data_file)
    new_yhat = diagnostics.model_predictions(dataset)
    
    _, _, new_y, _ = training.split_data(dataset)
    new_score = metrics.f1_score(new_y, new_yhat)

    logging.info(f'Latest score: {latest_score}, New score: {new_score}')

    if new_score <= latest_score:
        logging.info('No model drift')
        return False
    else:
        return True

def retrain_model(move_to_next_step):
    # Train new model if there's model drift
    if move_to_next_step == True:
        logging.info('training new model')
        training.main()
        scoring.score_model()
        return True


def redeploy_model(move_to_next_step):
    # Redeploy model if new one has been trained
    if move_to_next_step == True:
        logging.info('deploying new model')
        deployment.store_model_into_pickle()
        return True


def run_diagnostics_and_reporting(move_to_next_step):
    # Run diagnostics.py and reporting.py for the re-deployed model
    if move_to_next_step == True:
        logging.info('producing reporting and calling apis for statistics')
        reporting.score_model()
        subprocess.run(['python', 'apicalls.py'], check=True)
        return True



def main():
    with open('config.json', 'r') as f:
        config = json.load(f)

    input_folder_path = config['input_folder_path']
    output_folder_path = config['output_folder_path']
    prod_deployment_path = config['prod_deployment_path']
    model_path = config['output_model_path']

    new_files = check_new_data(input_folder_path, prod_deployment_path)
    move_to_next_step = ingest_new_data(new_files, input_folder_path, output_folder_path)
    
    if move_to_next_step:
        move_to_next_step = check_model_drift(move_to_next_step, output_folder_path, prod_deployment_path)
        move_to_next_step = retrain_model(move_to_next_step)
        move_to_next_step = redeploy_model(move_to_next_step)
        move_to_next_step = run_diagnostics_and_reporting(move_to_next_step)

if __name__ == '__main__':
    main()
