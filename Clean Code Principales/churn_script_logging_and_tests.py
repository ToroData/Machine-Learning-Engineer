"""
Set of unity tests for churn_library.py

Author: Ricard Santiago Raigada García
Date: January 2024
"""

import os
import logging
from churn_library import import_data, perform_eda, train_models
from churn_library import encoder_helper, perform_feature_engineering

logging.basicConfig(
    filename='./logs/churn_library.log',
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')

DATAFRAME = None
X_TRAIN = None
X_TEST = None
Y_TRAIN = None
Y_TEST = None

def test_import(import_data):
    '''
    Test the data import function.

    Args:
        import_data (function): A function to import data.

    Modifies:
        DATAFRAME (global variable): Set to the DataFrame loaded from the CSV file.

    Raises:
        FileNotFoundError: If the CSV file is not found.
        AssertionError: If the loaded DataFrame is empty.
    '''
    global DATAFRAME
    try:
        DATAFRAME = import_data("./data/bank_data.csv")
        logging.info("Testing import_data: SUCCESS")
    except FileNotFoundError as err:
        logging.error("Testing import_eda: The file wasn't found")
        raise err

    try:
        assert DATAFRAME.shape[0] > 0
        assert DATAFRAME.shape[1] > 0
    except AssertionError as err:
        logging.error(
            "Testing import_data: The file doesn't appear to have rows and columns")
        raise err


def test_eda(perform_eda):
    '''
    Test the perform EDA (Exploratory Data Analysis) function.

    Args:
        perform_eda (function): A function to perform EDA on a DataFrame.

    Modifies:
        Creates various EDA images in the specified file paths.

    Raises:
        Exception: If any error occurs during EDA process.
    '''
    global DATAFRAME
    try:
        perform_eda(DATAFRAME)
        # Check if EDA images are created
        assert os.path.isfile('images/EDA/churn_histogram.png')
        assert os.path.isfile('images/EDA/customer_age_histogram.png')
        logging.info("Testing perform_eda: SUCCESS")
    except Exception as err:
        logging.error(f"Testing perform_eda: {err}")
        raise err


def test_encoder_helper(encoder_helper):
    '''
    Test the encoder helper function.

    Args:
        encoder_helper (function): A function to encode categorical features.

    Modifies:
        DATAFRAME (global variable): Adds encoded feature columns to the DataFrame.

    Raises:
        AssertionError: If no new column is added to the DataFrame.
    '''
    global DATAFRAME
    try:
        original_shape = DATAFRAME.shape[1]
        DATAFRAME = encoder_helper(DATAFRAME, 'Marital_Status', 'Churn')
        # Check if new column added
        assert DATAFRAME.shape[1] == original_shape + 1
        logging.info("Testing encoder_helper: SUCCESS")
    except AssertionError as err:
        logging.error("Testing encoder_helper: New column not added")
        raise err


def test_perform_feature_engineering(perform_feature_engineering):
    '''
    Test the perform feature engineering function.

    Args:
        perform_feature_engineering (function): A function to perform feature engineering.

    Modifies:
        X_TRAIN, X_TEST, Y_TRAIN, Y_TEST (global variables): Splits the data into 
        training and test sets.

    Raises:
        AssertionError: If the data is not split correctly.
    '''
    global DATAFRAME, X_TRAIN, X_TEST, Y_TRAIN, Y_TEST
    try:
        X_TRAIN, X_TEST, Y_TRAIN, Y_TEST = perform_feature_engineering(
            DATAFRAME, 'Churn')
        # Check if data is split correctly
        assert X_TRAIN.shape[0] > 0
        assert X_TEST.shape[0] > 0
        assert Y_TRAIN.shape[0] > 0
        assert Y_TEST.shape[0] > 0
        logging.info("Testing perform_feature_engineering: SUCCESS")
    except AssertionError as err:
        logging.error("Testing perform_feature_engineering: Data split error")
        raise err


def test_train_models(train_models):
    '''
    Test the train models function.

    Args:
        train_models (function): A function to train models and save results.

    Checks:
        Verifies the creation of model files and plots.

    Raises:
        AssertionError: If model files or plots are not saved correctly.
    '''
    global X_TRAIN, X_TEST, Y_TRAIN, Y_TEST
    try:
        train_models(X_TRAIN, X_TEST, Y_TRAIN, Y_TEST)
        # Check if models and plots are saved
        assert os.path.isfile('./models/rfc_model.pkl')
        assert os.path.isfile('./models/logistic_model.pkl')
        assert os.path.isfile('./images/train/shap_summary_plot.png')
        assert os.path.isfile('./images/train/lr_roc_curve.png')
        assert os.path.isfile('./images/train/lr_roc_curve_TP_FP.png')
        assert os.path.isfile('./images/train/lr_classification_report.png')
        assert os.path.isfile('./images/train/rf_classification_report.png')
        logging.info("Testing train_models: SUCCESS")
    except AssertionError as err:
        logging.error("Testing train_models: Model or plot not saved")
        raise err


if __name__ == "__main__":
    test_import(import_data)
    test_eda(perform_eda)
    test_encoder_helper(encoder_helper)
    test_perform_feature_engineering(perform_feature_engineering)
    test_train_models(train_models)
