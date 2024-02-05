"""
Model and Data Diagnostics

This script performs diagnostic tests related to the deployed model and the input data.
It includes functions to:
    1. Get predictions made by the deployed model for a given dataset.
    2. Calculate summary statistics (means, medians, and standard deviations) for each
        numeric column in the input data.
    3. Check for missing data (NA values) in the input dataset and calculate the percentage
        of missing values for each column.
    4. Measure the execution time for data ingestion and model training tasks.
    5. Check and display outdated dependencies in the project.

Modules:
    - os: Provides operating system-related functionalities.
    - json: Provides JSON file handling.
    - pandas: Provides data manipulation capabilities.
    - numpy: Offers numerical operations and arrays.
    - timeit: Allows timing of code execution.
    - pip: Python package manager for checking dependencies.

Functions:
    - model_predictions(): Returns predictions made by the deployed model for
        a given dataset.
    - dataframe_summary(): Calculates summary statistics for each numeric column
        in the input data.
    - execution_time(): Measures execution time for data ingestion and model
        training tasks.
    - outdated_packages_list(): Checks and displays outdated Python package dependencies.

Author: Ricard Santiago Raigada García
Date: January, 2024
"""
import os
import json
import timeit
import pickle
import logging
import sys
import subprocess
import pandas as pd
from datetime import datetime
from training import split_data

# Load config.json and get environment variables
logging.basicConfig(stream=sys.stdout, level=logging.INFO)

with open('config.json', 'r') as f:
    config = json.load(f)

dataset_csv_path = os.path.join(config['output_folder_path'])
test_data_path = os.path.join(config['test_data_path'])
prod_deployment_path = os.path.join(config['prod_deployment_path'])


def model_predictions() -> list:
    """
    Get predictions made by the deployed model for a given dataset.

    Args:
        dataset (pd.DataFrame): The input dataset in pandas DataFrame format.

    Returns:
        list: A list containing all predictions made by the deployed model.
    """
    logging.info("Running model_predictions function")
    # Load the dataset from the production deployment directory
    try:
        dataset = pd.read_csv(os.path.join(test_data_path, 'testdata.csv'))
        logging.info("Dataset loaded successfully")
    except Exception as e:
        return logging.error(f"Can't load the dataset. ERROR: {e}")

    # Load the deployed model from the production deployment directory
    try:
        model_path = os.path.join(prod_deployment_path, 'trainedmodel.pkl')
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        logging.info("Model loaded successfully")
    except Exception as e:
        return logging.error(f"Can't load the model. ERROR: {e}")

    # Split data and predict
    logging.info("Running predictions")
    X_train, X_test, y_train, y_test = split_data(dataset)
    pred = model.predict(X_train)
    logging.info(f"Predictions: {pred}")

    return pred


def dataframe_summary() -> list:
    """
    Calculate summary statistics (means, medians, and standard deviations) for each
    numeric column in the input data.

    Args:
        dataset (pd.DataFrame): The input dataset in pandas DataFrame format.

    Returns:
        list: A list containing all summary statistics for every numeric
            column of the input dataset.
    """
    # Load dataset
    dataset = pd.read_csv(os.path.join(dataset_csv_path, 'finaldata.csv'))

    # Calculate summary statistics for each numeric column
    logging.info("Running dataframe_summary function")
    summary_statistics = dataset.describe()
    logging.info(f"Summary statistics: {summary_statistics.values.tolist()}")

    return summary_statistics.values.tolist()


def missing_data() -> list:
    """
    Check and calculate the percentage of missing data (NA values) in each column of the dataset.

    Returns:
        list: A list containing the percentage of missing data for each column.
    """
    # Load the dataset
    dataset = pd.read_csv(os.path.join(dataset_csv_path, 'finaldata.csv'))

    # Calculate the percentage of missing data for each column
    missing_percentages = (dataset.isnull().sum() / len(dataset)) * 100

    # Convert the missing percentages to a list
    missing_percentages_list = missing_percentages.tolist()

    # Results
    logging.info(f"Missing values: {missing_percentages_list}")
    return missing_percentages_list


def execution_time() -> list:
    """
    Measure the execution time for data ingestion and model training tasks.

    Returns:
        list: A list of 2 timing values in seconds: data ingestion time and model training time.
    """
    # Timing data ingestion task
    data_ingestion_time = timeit.timeit(
        "subprocess.run(['python', 'ingestion.py'], stdout=subprocess.PIPE)",
        setup="import subprocess",
        number=1,
    )

    # Timing model training task
    model_training_time = timeit.timeit(
        "subprocess.run(['python', 'training.py'], stdout=subprocess.PIPE)",
        setup="import subprocess",
        number=1,
    )

    # Store execution times with timestamps
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    execution_times = {
        "data_ingestion_time": data_ingestion_time,
        "model_training_time": model_training_time,
    }

    # Save execution times to files with timestamps
    execution_dir = os.path.join("olddiagnostics", timestamp)
    os.makedirs(execution_dir, exist_ok=True)

    for task, time in execution_times.items():
        with open(os.path.join(execution_dir, f"{task}_{timestamp}.txt"), "w") as f:
            f.write(f"{task}: {time} seconds")

    logging.info(f"Data Ingestion Time: {data_ingestion_time}")
    logging.info(f"Model Training Time: {model_training_time}")

    return [data_ingestion_time, model_training_time]


def outdated_packages_list() -> str:
    """
    Check and display outdated Python package dependencies in the project.

    Returns:
        str: A string containing information about outdated packages.
    """
    # Run a pip command to check outdated packages
    try:
        result = subprocess.run(
            ['pip', 'list', '--outdated', '--format=json'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        outdated_packages_info = result.stdout
    except Exception as e:
        outdated_packages_info = str(e)

    logging.info(outdated_packages_info)

    return outdated_packages_info


if __name__ == '__main__':
    model_predictions()
    dataframe_summary()
    missing_data()
    execution_time()
    outdated_packages_list()
