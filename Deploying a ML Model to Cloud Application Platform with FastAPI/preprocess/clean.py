"""
Module to preprocess input dataset

Author: Ricard Santiago Raigada García
Date: February, 2024
"""

import json
import logging
import sys
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer


# Logging configuration
logging.basicConfig(stream=sys.stdout, level=logging.INFO)

# Load config.json and get input and output paths
with open('./config.json', 'r') as f:
    config = json.load(f)

input_data = config['input_data']
output_clean_data = config['output_clean_data']


def process_data(file_path: str, save_path: str) -> None:
    """Perform the clean

    Args:
        file_path (str): path to the dataset
        save_path (str): path to the output folder and name of the file
    """
    # Load data
    data = pd.read_csv(file_path)

    # Adjust column names by removing leading and trailing whitespace
    data.columns = data.columns.str.strip()

    # Save
    data.to_csv(save_path, index=False)

    logging.info(f'Dataset clean saved in {save_path}')


if __name__ == '__main__':
    logging.info("Running clean.py")
    process_data(file_path=input_data, save_path=output_clean_data)
    logging.info("Successful cleaning")
