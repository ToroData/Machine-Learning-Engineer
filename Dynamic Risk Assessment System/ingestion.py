import os
import json
from datetime import datetime
import logging
import sys
import pandas as pd


logging.basicConfig(stream=sys.stdout, level=logging.INFO)

# Load config.json and get input and output paths
with open('config.json', 'r') as f:
    config = json.load(f)

input_folder_path = config['input_folder_path']
output_folder_path = config['output_folder_path']


def list_csv_files(input_folder_path: str) -> list:
    """
    List all CSV files in the input folder.

    Args:
        input_folder_path (str): The path to the input folder.

    Returns:
        List[str]: A list of CSV file names in the input folder.
    """
    csv_files = [file for file in os.listdir(
        input_folder_path) if file.endswith('.csv')]
    return csv_files


def read_csv_files(input_folder_path: str, csv_files: list) -> list:
    """
    Read CSV files from the input folder into Pandas DataFrames.

    Args:
        input_folder_path (str): The path to the input folder.
        csv_files (list): A list of CSV file names.

    Returns:
        List[pd.DataFrame]: A list of Pandas DataFrames, one for each CSV file.
    """
    dataframes = []
    for file in csv_files:
        file_path = os.path.join(input_folder_path, file)
        df = pd.read_csv(file_path)
        dataframes.append(df)
    return dataframes


def concatenate_dataframes(dataframes: list) -> pd.DataFrame:
    """
    Concatenate a list of Pandas DataFrames into one master DataFrame.

    Args:
        dataframes (List[pd.DataFrame]): A list of Pandas DataFrames.

    Returns:
        pd.DataFrame: The concatenated master DataFrame.
    """
    logging.info("Concatenate all DataFrames")
    master_df = pd.concat(dataframes, ignore_index=True)
    return master_df


def remove_duplicates(master_df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove duplicate rows from a DataFrame.

    Args:
        master_df (pd.DataFrame): The input DataFrame.

    Returns:
        pd.DataFrame: The DataFrame with duplicates removed.
    """
    logging.info("Dropping duplicates")
    master_df.drop_duplicates(inplace=True)
    return master_df


def create_output_folder(output_folder_path: str) -> None:
    """
    Create the output directory if it doesn't exist.

    Args:
        output_folder_path (str): The path to the output folder.
    """
    os.makedirs(output_folder_path, exist_ok=True)


def save_master_dataframe(
        master_df: pd.DataFrame,
        output_folder_path: str) -> None:
    """
    Save a DataFrame as finaldata.csv in the output folder.

    Args:
        master_df (pd.DataFrame): The DataFrame to be saved.
        output_folder_path (str): The path to the output folder.
    """
    logging.info("Saving ingested data")
    output_file_path = os.path.join(output_folder_path, 'finaldata.csv')
    master_df.to_csv(output_file_path, index=False)


def save_ingested_files_metadata(
        csv_files: list,
        output_folder_path: str) -> None:
    """
    Save metadata about ingested files as ingestedfiles.txt.

    Args:
        csv_files (list): A list of CSV file names.
        output_folder_path (str): The path to the output folder.
    """
    # List of the filenames of all ingested files
    ingested_files = [file for file in csv_files]

    # Save the list of ingested files as ingestedfiles.txt
    logging.info("Saving ingested metadata")
    ingested_files_path = os.path.join(output_folder_path, 'ingestedfiles.txt')
    with open(ingested_files_path, 'w') as ingested_file:
        for file in ingested_files:
            ingested_file.write(
                f"{file}:\tIngestion date: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}\n")


def merge_multiple_dataframe(
        input_folder_path: str,
        output_folder_path: str) -> None:
    """
    Perform data ingestion from CSV files in the input folder.

    Args:
        input_folder_path (str): The path to the input folder.
        output_folder_path (str): The path to the output folder.
    """
    csv_files = list_csv_files(input_folder_path)
    dataframes = read_csv_files(input_folder_path, csv_files)
    master_df = concatenate_dataframes(dataframes)
    master_df = remove_duplicates(master_df)
    create_output_folder(output_folder_path)
    save_master_dataframe(master_df, output_folder_path)
    save_ingested_files_metadata(csv_files, output_folder_path)


if __name__ == '__main__':
    logging.info("Running ingestion.py")
    merge_multiple_dataframe(input_folder_path, output_folder_path)
    logging.info("Successful ingestion")
