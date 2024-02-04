"""
Dynamic Risk Assessment System - Model Training Script

This script is responsible for training a logistic regression model for dynamic risk assessment.
It reads data from a CSV file, splits it into training and testing sets, optimizes hyperparameters
using Optuna, trains the model, and saves it to a file.

Modules:
    - pandas: Provides data manipulation capabilities.
    - numpy: Offers numerical operations and arrays.
    - pickle: Allows for serialization of Python objects.
    - os: Provides operating system-related functionalities.
    - sklearn: Scikit-learn library for machine learning tools.
    - json: Provides JSON file handling.
    - logging: Enables logging for debugging and monitoring.
    - sys: Provides access to system-specific parameters and functions.
    - optuna: A library for hyperparameter optimization.

Functions:
    - read_data(filepath): Reads data from a CSV file and returns a pandas DataFrame.
    - split_data(data): Splits data into features and target variable for training and testing.
    - evaluate_model(model, X_test, y_test): Evaluates model accuracy on testing data.
    - objective(trial): Defines the objective function for Optuna hyperparameter optimization.
    - optuna_study(objective): Performs hyperparameter optimization using Optuna.
    - train_model(X_train, X_test, y_train, y_test, hyperparameters): Trains a LR model.
    - save_model(model, output_model_path, model_path): Saves the trained model to a file.
    - main(): The main function that orchestrates the model training process.

This script uses configuration data from 'config.json' to determine input and output paths,
as well as initial hyperparameters for the model.

Author: Ricard Santiago Raigada García
Date: January, 2024
"""
import json
import logging
import sys
import os
import pickle
import optuna
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
import pandas as pd
from ingestion import create_output_folder

# Logging config
logging.basicConfig(stream=sys.stdout, level=logging.INFO)

# Load config.json and get path variables
with open('config.json', 'r') as f:
    config = json.load(f)

dataset_csv_path = os.path.join(config['output_folder_path'])
model_path = os.path.join(config['output_model_path'])


def read_data(filepath: str) -> pd.DataFrame:
    """
    Read the data from CSV.

    Returns:
        pd.DataFrame: The data read from the CSV file.
    """
    logging.info(f"Loading dataset from {filepath}")
    data = pd.read_csv(filepath)
    return data


def split_data(data: pd.DataFrame) -> pd.DataFrame:
    """
    Split data into features and the target variable.

    Args:
        data (pd.DataFrame): The input data.

    Returns:
        pd.DataFrame, pd.Series: Features (X) and target variable (y).
    """
    logging.info("Splitting data...")
    X = data[['lastmonth_activity', 'lastyear_activity', 'number_of_employees']]
    y = data['exited']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=0)
    return X_train, X_test, y_train, y_test


def evaluate_model(
        model: LogisticRegression,
        X_test: pd.DataFrame,
        y_test: pd.Series) -> float:
    """
    Evaluate the model's accuracy on the testing data.

    Args:
        model (sklearn.linear_model.LogisticRegression): Trained model.
        X_test (pd.DataFrame): Features for testing.
        y_test (pd.Series): Target variable for testing.

    Returns:
        float: Model accuracy.
    """
    logging.info(f"Evaluation model...")
    y_pred = model.predict(X_test)
    accuracy = metrics.accuracy_score(y_test, y_pred)
    return accuracy


def objective(trial: optuna.trial.Trial):
    """
    Objective function to optimize hyperparameters for a logistic regression model using Optuna.

    Args:
        trial (optuna.trial.Trial): The current Optuna trial.

    Returns:
        float or None: The accuracy score obtained during cross-validation,
            or None if an exception occurs.
    """
    # Some parameters for fine-tunning
    # C = trial.suggest_loguniform('C', 1e-3, 1e3)
    # class_weight = trial.suggest_categorical('class_weight', [None, 'balanced'])
    # dual = trial.suggest_categorical('dual', [False, True])
    # fit_intercept = trial.suggest_categorical('fit_intercept', [False, True])
    # intercept_scaling = trial.suggest_loguniform('intercept_scaling', 1e-3, 1e3)
    # l1_ratio = trial.suggest_categorical('l1_ratio', [None, 0.25, 0.5, 0.75])
    # max_iter = trial.suggest_int('max_iter', 50, 1000)
    # multi_class = trial.suggest_categorical('multi_class', ['auto', 'ovr', 'multinomial'])
    # n_jobs = trial.suggest_categorical('n_jobs', [None, -1, 1, 2, 4])
    # penalty = trial.suggest_categorical('penalty', ['l1', 'l2'])
    # random_state = trial.suggest_int('random_state', 0, 100)
    # solver = trial.suggest_categorical('solver', [
    # 'newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'])
    # tol = trial.suggest_loguniform('tol', 1e-6, 1e-2)
    # verbose = trial.suggest_categorical('verbose', [0, 1, 2])
    # warm_start = trial.suggest_categorical('warm_start', [False, True])

    params = {
        'tol': trial.suggest_uniform('tol', 1e-6, 1e-3),
        'C': trial.suggest_loguniform("C", 1e-2, 1),
        'random_state': trial.suggest_categorical('random_state', [0, 42, 2021, 555]),
        "n_jobs": -1
    }

    # Load data
    dataset = read_data(os.path.join(dataset_csv_path, "finaldata.csv"))
    X_train, X_test, y_train, y_test = split_data(dataset)

    # Create and evaluate a logistic regression model with hyperparameters
    model = LogisticRegression(**params)

    # Evaluate the model using cross validation
    try:
        # Cross Validation
        scores = cross_val_score(
            model, X_train, y_train, cv=5, scoring='accuracy')
        accuracy = scores.mean()
    except Exception as e:
        return e

    return accuracy


def optuna_study(objective) -> None:
    """
    Perform an Optuna study to optimize hyperparameters for a logistic regression model.

    Args:
        objective (function): The objective function to optimize.

    Returns:
        None
    """
    logging.info(f"Creating Optuna Study")
    # Create an Optuna studio
    study = optuna.create_study(
        direction='minimize',
        study_name='lr',
        pruner=optuna.pruners.HyperbandPruner())

    # Optimize hyperparameters
    study.optimize(objective, n_trials=300)

    # Get the best hyperparameters
    best_hyperparameters = study.best_params

    with open('config.json', 'r') as config_file:
        config_data = json.load(config_file)

    config_data['hyperparameters'] = best_hyperparameters

    # Save updated configuration in config.json
    with open('config.json', 'w') as config_file:
        json.dump(config_data, config_file, indent=4)

    logging.info(
        "Updated configuration in config.json with better hyperparameters.")


def train_model(
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series,
        hyperparameters: dict) -> LogisticRegression:
    """
    Train a logistic regression model on the data and save it.

    Args:
        X_train (pd.DataFrame): Training features.
        X_test (pd.DataFrame): Testing features.
        y_train (pd.Series): Training target variable.
        y_test (pd.Series): Testing target variable.
        hyperparameters (dict): Hyperparameters for the logistic regression model.

    Returns:
        sklearn.linear_model.LogisticRegression: Trained model.
    """
    logging.info(f"Training model with configuration: {hyperparameters}")
    model = LogisticRegression(**hyperparameters)
    model.fit(X_train, y_train)

    accuracy = evaluate_model(model, X_test, y_test)
    logging.info(f"Model trained with accuracy: {accuracy:.2f}")

    return model


def save_model(
        model: LogisticRegression,
        output_model_path: str,
        model_path: str) -> None:
    """
    Save the trained model to a file.

    Args:
        model (sklearn.linear_model.LogisticRegression): Trained model.
        output_model_path (str): Output model path.
        model_path (str): Full path to save the model.
    """
    create_output_folder(output_model_path)
    logging.info(f"Saving {model_path}")
    with open(model_path, 'wb') as model_file:
        pickle.dump(model, model_file)


def main():
    """
    Main function to train a logistic regression model and save it.
    Loads the dataset, optimizes hyperparameters using Optuna, and saves the trained model.
    """
    dataset = read_data(os.path.join(dataset_csv_path, "finaldata.csv"))
    X_train, X_test, y_train, y_test = split_data(dataset)

    # Best hyperparameters with Optuna
    optuna_study(objective)
    hyperparameters = config['hyperparameters']

    model = train_model(X_train, X_test, y_train, y_test, hyperparameters)
    save_model(
        model_path,
        config['output_model_path'],
        os.path.join(
            model_path,
            'trainedmodel.pkl'))
    logging.info(f"Model training completed.")


if __name__ == '__main__':
    main()
