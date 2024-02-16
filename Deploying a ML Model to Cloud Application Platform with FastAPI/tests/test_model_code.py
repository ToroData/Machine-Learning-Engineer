import pytest
import json
import pandas as pd
from sklearn.model_selection import train_test_split
from ml.train_model import process_data
from ml.model import train_model, compute_model_metrics, inference
import numpy as np
import logging

@pytest.fixture(scope="module")
def data():
    with open('./config.json', 'r') as f:
        config = json.load(f)
    output_clean_data = config['output_clean_data']
    return pd.read_csv(output_clean_data)

@pytest.fixture(scope="module")
def preprocessed_data(data):
    cat_features = [
        "workclass", "education", "marital-status", "occupation",
        "relationship", "race", "sex", "native-country",
    ]
    logging.info("Processing the training data")
    train, _ = train_test_split(data, test_size=0.20, random_state=10, stratify=data['salary'])
    X_train, y_train, encoder, lb = process_data(
        train, categorical_features=cat_features, label="salary", training=True)
    return X_train, y_train, encoder, lb

@pytest.fixture(scope="module")
def model_and_metrics(preprocessed_data):
    X_train, y_train, _, _ = preprocessed_data
    model = train_model(X_train, y_train)
    preds = inference(model, X_train)
    metrics = compute_model_metrics(y_train, preds)
    return model, metrics

def test_train_model(preprocessed_data):
    X_train, y_train, _, _ = preprocessed_data
    model = train_model(X_train, y_train)
    assert model is not None

def test_compute_model_metrics(model_and_metrics):
    _, metrics = model_and_metrics
    precision, recall, fbeta = metrics
    assert 0 <= precision <= 1
    assert 0 <= recall <= 1
    assert 0 <= fbeta <= 1

def test_inference(model_and_metrics, preprocessed_data):
    model, _ = model_and_metrics
    X_train, _, _, _ = preprocessed_data
    preds = inference(model, X_train)
    assert len(preds) == len(X_train)
    assert all(pred in [0, 1] for pred in preds)
