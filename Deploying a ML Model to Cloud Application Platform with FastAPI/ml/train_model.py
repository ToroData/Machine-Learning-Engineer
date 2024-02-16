"""# Script to train machine learning model.
Author: Ricard Santiago Raigada García
Date: February, 2024
"""
import os
import json
import pickle
import sys
import logging
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer, OneHotEncoder
from ml.model import train_model, compute_model_metrics, inference
from preprocess.clean import process_data

# Logging configuration
logging.basicConfig(stream=sys.stdout, level=logging.INFO)

# Load config.json and get input and output paths
with open('./config.json', 'r') as f:
    config = json.load(f)

output_clean_data = config['output_clean_data']
output_model_path = config['output_model_path']

def process_data(
    X, 
    categorical_features=[], 
    label=None, 
    training=True, 
    encoder=None, 
    lb=None
):
    """
    Process the data used in the machine learning pipeline.

    Args:
    - X (pd.DataFrame): Dataframe containing the features and label.
    - categorical_features (list[str]): List of column names to be treated as categorical features.
    - label (str): Name of the label column in `X`. If None, then an empty array will be returned for y.
    - training (bool): Indicates whether the function is being called during training.
    - encoder (OneHotEncoder): Pre-fitted OneHotEncoder instance if training=False.
    - lb (LabelBinarizer): Pre-fitted LabelBinarizer instance if training=False.

    Returns:
    - X (np.array): Processed data.
    - y (np.array): Processed labels if `label` is not None, otherwise an empty array.
    - encoder (OneHotEncoder): Fitted OneHotEncoder instance if training=True.
    - lb (LabelBinarizer): Fitted LabelBinarizer instance if training=True and `label` is not None.
    """
    if label is not None:
        y = X[label]
        X = X.drop([label], axis=1)
    else:
        y = np.array([])

    X_categorical = X[categorical_features].values
    X_continuous = X.drop(*[categorical_features], axis=1)

    if training is True:
        encoder = OneHotEncoder(sparse=False, handle_unknown="ignore")
        lb = LabelBinarizer()
        X_categorical = encoder.fit_transform(X_categorical)
        y = lb.fit_transform(y.values).ravel()
    else:
        X_categorical = encoder.transform(X_categorical)
        try:
            y = lb.transform(y.values).ravel()
        # Catch the case where y is None because we're doing inference.
        except AttributeError:
            pass

    X = np.concatenate([X_continuous, X_categorical], axis=1)
    logging.info("Data processed successfully")
    return X, y, encoder, lb

# Load the data
logging.info("Importing data")
data = pd.read_csv(output_clean_data)

# Split the data into training and test sets
logging.info("Splitting...")
train, test = train_test_split(data, 
                                test_size=0.20, 
                                random_state=10, 
                                stratify=data['salary']
                                )
# Define the categorical features for the encoding
cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]

# Process the training data
logging.info("Processing the training data")
X_train, y_train, encoder, lb = process_data(
    train,
    categorical_features=cat_features,
    label="salary",
    training=True
)

# Process the test data
logging.info("Processing the test data")
X_test, y_test, encoder, lb = process_data(
    test,
    categorical_features=cat_features,
    label="salary",
    training=False,
    encoder=encoder,
    lb=lb
)

# Train and save the model
logging.info("Training...")
model = train_model(X_train, y_train)

model_filename = "model.pkl"
encoder_filename = "encoder.pkl"
lb_filename = "lb.pkl"

# Save model on disk
logging.info("Saving model...")
pickle.dump(model, open(os.path.join(output_model_path, model_filename), 'wb'))
logging.info(f"Model saved to disk: {os.path.join(output_model_path, model_filename)}")

# Save encoder and lb
pickle.dump(encoder, open(os.path.join(output_model_path, encoder_filename), 'wb'))
pickle.dump(lb, open(os.path.join(output_model_path, lb_filename), 'wb'))
logging.info(f"Encoder and LabelBinarizer saved to disk: {output_model_path}")

# Make predictions on the test set
logging.info("Predicting")
preds = inference(model, X_test)

# Calculate metrics
precision, recall, fbeta = compute_model_metrics(y_test, preds)

print(f"Precision: {precision}, Recall: {recall}, F-beta: {fbeta}")
