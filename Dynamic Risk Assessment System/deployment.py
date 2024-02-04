"""
Model Deployment Script

This script is responsible for deploying the trained model and related files to
the production deployment directory.
It copies the trained model (trainedmodel.pkl), model score (latestscore.txt),
and ingested data record (ingestedfiles.txt)
to the production deployment directory specified in the 'config.json' file.

Modules:
    - os: Provides operating system-related functionalities.
    - shutil: Provides file operations for copying files.
    - json: Provides JSON file handling.

Functions:
    - deploy_model(): Copies the necessary files to the production deployment directory.

Author: Ricard Santiago Raigada García
Date: January, 2024
"""
import os
import shutil
import json
import logging
import sys

logging.basicConfig(stream=sys.stdout, level=logging.INFO)

with open('config.json', 'r') as f:
    config = json.load(f)

dataset_csv_path = os.path.join(config['output_folder_path'])
prod_deployment_path = os.path.join(config['prod_deployment_path'])
model_path = os.path.join(config['output_model_path'])
output_folder_path = config['output_folder_path']


def store_model_into_pickle() -> None:
    """
    Copy the trained model, model score, and ingested data record to the production
    deployment directory.

    Returns:
        None
    """
    # Ensure the production deployment directory exists
    if not os.path.exists(prod_deployment_path):
        os.makedirs(prod_deployment_path)

    try:
        # Copy the trained model (trainedmodel.pkl) to the production deployment
        # directory
        shutil.copy(
            os.path.join(
                model_path,
                'trainedmodel.pkl'),
            prod_deployment_path)

        # Copy the model score (latestscore.txt) to the production deployment
        # directory
        shutil.copy(
            os.path.join(
                model_path,
                'latestscore.txt'),
            prod_deployment_path)

        # Copy the ingested data record (ingestedfiles.txt) to the production
        # deployment directory
        shutil.copy(
            os.path.join(
                output_folder_path,
                'ingestedfiles.txt'),
            prod_deployment_path)

        logging.info("Copied successfully")
    except Exception as e:
        logging.error(f"Error during deployment: {str(e)}")


if __name__ == '__main__':
    store_model_into_pickle()
