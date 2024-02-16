import sys
import logging
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import fbeta_score, precision_score, recall_score


# Optional: implement hyperparameter tuning.
def train_model(X_train, y_train):
    """
    Trains a machine learning model using RandomForest and returns it.
    Implements automatic hyperparameter tuning using GridSearchCV with two threads.

    Inputs
    ------
    X_train : np.array
        Training data.
    y_train : np.array
        Labels.

    Returns
    -------
    best_model : RandomForestClassifier
        The best model found by GridSearchCV.
    """
    # RandomForestClassifier
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    }

    # Initialize the RandomForestClassifier
    logging.info("Initialize the RandomForestClassifier")
    rf = RandomForestClassifier(random_state=42)

    # Initialize GridSearchCV with two threads (n_jobs=2)
    logging.info("Initialize GridSearchCV with two threads (n_jobs=2)")
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, n_jobs=2, scoring='accuracy')

    # Fit GridSearchCV to the training data
    logging.info("Fit GridSearchCV to the training data")
    grid_search.fit(X_train, y_train)

    # Get the best model
    logging.info("Get the best model")
    best_model = grid_search.best_estimator_

    logging.info(f"Best parameters found: {grid_search.best_params_}")
    logging.info(f"Best accuracy found: {grid_search.best_score_}")

    return best_model


def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using precision, recall, and F1.

    Inputs
    ------
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    logging.info("Compute the metrics")
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta


def inference(model, X):
    """
    Run model inferences and return the predictions.

    Inputs
    ------
    model : RandomForestClassifier
        Trained machine learning model.
    X : np.array
        Data used for prediction.

    Returns
    -------
    preds : np.array
        Predictions from the model.
    """
    logging.info("Running inference...")
    preds = model.predict(X)
    return preds


def compute_confusion_matrix(y, preds, labels=None):
    """
    Compute confusion matrix using the predictions and ground thruth provided
    Inputs
    ------
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.
    Returns
    ------
    cm : confusion matrix for the provided prediction set
    """
    logging.info("Computing confusion matrix...")
    cm = confusion_matrix(y, preds)
    return cm


def evaluate_model_on_slices(model, X, y, categorical_features, encoder):
    """
    Evaluate the performance of the model on slices of the data based on categorical features.

    Args:
    - model: The trained machine learning model.
    - X: np.array, the features of the dataset.
    - y: np.array, the target labels.
    - categorical_features: list of str, the names of the categorical features.
    - encoder: OneHotEncoder, the trained OneHotEncoder instance used on the categorical features.

    Returns:
    - None, but prints out the performance metrics for each slice of the data.
    """
    logging.info("Evaluating model on slices...")
    # Get the feature names after one-hot encoding
    feature_names = encoder.get_feature_names_out(categorical_features)
    
    for feature in categorical_features:
        # Find the indices for the categories of the current feature
        feature_indices = [i for i, name in enumerate(feature_names) if name.startswith(f"{feature}_")]
        
        for idx in feature_indices:
            category = feature_names[idx].split('_')[-1]
            subset_mask = X[:, idx] == 1
            X_subset = X[subset_mask]
            y_subset = y[subset_mask]
            
            if len(y_subset) > 0:  # Ensure there are samples in the subset
                preds_subset = inference(model, X_subset)
                precision, recall, fbeta = compute_model_metrics(y_subset, preds_subset)
                print(f"Performance on slice '{feature}={category}': Precision={precision}, Recall={recall}, F-beta={fbeta}")
            else:
                print(f"No samples for slice '{feature}={category}'.")
