"""
This module is responsible for scoring a trained machine learning model.
It loads a trained model from disk, uses it to make predictions on test data,
calculates the F1 score, and saves the score to a file.

Modules:
    - pandas: Provides data manipulation capabilities.
    - pickle: Allows for serialization of Python objects.
    - os: Provides operating system-related functionalities.
    - sys: Provides access to system-specific parameters and functions.
    - json: Provides JSON file handling.
    - logging: Enables logging for debugging and monitoring.
    - sklearn.metrics: Scikit-learn library for machine learning metrics.
    - training: A custom module for splitting data.

Functions:
    - load_model(model_path: str): Loads a trained model from disk.
    - score_model(): Scores the loaded model on test data and saves the F1 score.

Author: Ricard Santiago Raigada García
Date: January, 2024
"""
import pickle
import os
import sys
import json
import logging
from sklearn.metrics import f1_score
import pandas as pd
from training import split_data

# Load config.json and get path variables
logging.basicConfig(stream=sys.stdout, level=logging.INFO)

with open('config.json', 'r') as f:
    config = json.load(f)

dataset_csv_path = os.path.join(config['output_folder_path'])
test_data_path = os.path.join(config['test_data_path'])
model_path = os.path.join(config['output_model_path'])
testdata = pd.read_csv(os.path.join(test_data_path, 'testdata.csv'))


def load_model(model_path: str) -> object:
    """Loads the trained model from disk.

    Args:
        model_path (str): The path to the saved model file.

    Returns:
        object: The loaded model.

    Raises:
        FileNotFoundError: If the specified model file does not exist.
        EOFError: If there is an issue with deserializing the model.
        Exception: For any other unexpected errors during loading.
    """
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        logging.info(f"Model loaded successfully, type: {type(model)}")
        return model
    except FileNotFoundError as e:
        logging.error(f"File not found: {model_path}")
        raise
    except EOFError as e:
        logging.error(f"Error occurred while deserializing the model: {e}")
        raise
    except Exception as e:
        logging.error(
            f"An unexpected error occurred while loading the model: {e}")
        raise


def score_model() -> float:
    """
    Load a trained machine learning model, make predictions on test data,
    calculate the F1 score, and save the score to a file.

    Returns:
        float: The F1 score of the model on the test data.
    """
    # Load model
    modelpath = os.path.join(model_path, 'trainedmodel.pkl')
    model = load_model(modelpath)

    # Split data
    X_test, _, y_test, _ = split_data(testdata)

    # Predict using the loaded model
    y_pred = model.predict(X_test)

    # Calculate the F1 score
    logging.info("Evaluating model...")
    f1 = f1_score(y_test, y_pred)
    logging.info(f"F1 score: {f1}")

    # Write the F1 score to the latestscore.txt file
    score_file_path = os.path.join(model_path, 'latestscore.txt')
    with open(score_file_path, 'w') as score_file:
        score_file.write(str(f1))
    logging.info(f"F1 score saved to {score_file_path}")

    return f1


if __name__ == '__main__':
    score_model()
