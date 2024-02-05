"""
Full Process Script

This script performs a series of data processing and model-related tasks, including checking for new data, ingesting it,
detecting model drift, retraining the model, redeploying it, and running diagnostics and reporting.

It uses multiple functions to accomplish these tasks, which are organized as follows:
- `check_new_data`: Checks for new data files in the input folder.
- `ingest_new_data`: Ingests new data files into the output folder.
- `check_model_drift`: Checks for model drift and returns True if detected.
- `retrain_model`: Retrains the model if model drift is detected.
- `redeploy_model`: Redeploys the model if a new model is trained.
- `run_diagnostics_and_reporting`: Runs diagnostics and reporting for the re-deployed model.

This script can be run as a standalone program or imported as a module for reuse in other scripts or workflows.

Usage:
    To use this script, configure the 'config.json' file with the appropriate paths and settings,
    and then execute this script by running 'python fullprocess.py'.

Author: Ricard Santiago Raigada García
Date: January, 2024
"""
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


def check_new_data(input_folder_path, prod_deployment_path):
    """
    Check and read new data from the input_folder_path and compare it with ingestedfiles.txt in prod_deployment_path.

    Args:
        input_folder_path (str): Path to the input data folder.
        prod_deployment_path (str): Path to the production deployment folder.

    Returns:
        list: List of new data files found in the input folder.
    """
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
    """
    Ingest new data files into the output_folder_path by merging them if files are provided.

    Args:
        files (list): List of new data files to ingest.
        input_folder_path (str): Path to the input data folder.
        output_folder_path (str): Path to the output data folder.

    Returns:
        bool: True if new data was ingested, False otherwise.
    """
    # Ingest new data
    if files:
        ingestion.merge_multiple_dataframe(
            input_folder_path, output_folder_path)
        return True
    else:
        return False


def check_model_drift(
        move_to_next_step,
        output_folder_path,
        prod_deployment_path):
    """
    Check for model drift by comparing the latest score with the new score.

    Args:
        move_to_next_step (bool): Whether to move to the next step.
        output_folder_path (str): Path to the output data folder.
        prod_deployment_path (str): Path to the production deployment folder.

    Returns:
        bool: True if model drift is detected, False otherwise.
    """
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
    """
    Retrain the model if model drift is detected.

    Args:
        move_to_next_step (bool): Whether to move to the next step.

    Returns:
        bool: True if a new model is trained, False otherwise.
    """
    # Train new model if there's model drift
    if move_to_next_step:
        logging.info('training new model')
        training.main()
        scoring.score_model()
        return True


def redeploy_model(move_to_next_step):
    """
    Redeploy the model if a new model is trained.

    Args:
        move_to_next_step (bool): Whether to move to the next step.

    Returns:
        bool: True if a new model is deployed, False otherwise.
    """
    # Redeploy model if new one has been trained
    if move_to_next_step:
        logging.info('deploying new model')
        deployment.store_model_into_pickle()
        return True


def run_diagnostics_and_reporting(move_to_next_step):
    """
    Run diagnostics and reporting for the re-deployed model.

    Args:
        move_to_next_step (bool): Whether to move to the next step.

    Returns:
        bool: True if diagnostics and reporting are executed, False otherwise.
    """
    # Run diagnostics.py and reporting.py for the re-deployed model
    if move_to_next_step:
        logging.info('producing reporting and calling apis for statistics')
        reporting.score_model()
        subprocess.run(['python', 'apicalls.py'], check=True)
        return True


def main():
    """
    Main function to control the overall workflow.
    """
    with open('config.json', 'r') as f:
        config = json.load(f)

    input_folder_path = config['input_folder_path']
    output_folder_path = config['output_folder_path']
    prod_deployment_path = config['prod_deployment_path']
    model_path = config['output_model_path']

    new_files = check_new_data(input_folder_path, prod_deployment_path)
    move_to_next_step = ingest_new_data(
        new_files, input_folder_path, output_folder_path)

    if move_to_next_step:
        move_to_next_step = check_model_drift(
            move_to_next_step, output_folder_path, prod_deployment_path)
        move_to_next_step = retrain_model(move_to_next_step)
        move_to_next_step = redeploy_model(move_to_next_step)
        move_to_next_step = run_diagnostics_and_reporting(move_to_next_step)


if __name__ == '__main__':
    main()
