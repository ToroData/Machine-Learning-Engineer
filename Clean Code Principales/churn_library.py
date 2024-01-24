"""
Library to compute data science functions

Author: Ricard Santiago Raigada García
Date: January 2024
"""

# Import libraries
import os
import logging
import shap
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import plot_roc_curve, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

os.environ['QT_QPA_PLATFORM'] = 'offscreen'
sns.set()


# Logging configuration
logging.basicConfig(
    filename='./logs/churn_library.log',
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s'
)


def import_data(path: str) -> pd.DataFrame:
    """
    Returns dataframe for the csv found at path

    input:
            path: a path to the csv
    output:
            data_frame: pandas dataframe
    """
    logging.info("Importing data.")
    local_data_frame = pd.read_csv(path)
    local_data_frame['Churn'] = local_data_frame['Attrition_Flag'].apply(
        lambda val: 0 if val == "Existing Customer" else 1)
    return local_data_frame


def perform_eda(dataframe: pd.DataFrame) -> None:
    '''
    perform eda on dataframe and save figures to images folder
    input:
            dataframe: pandas dataframe

    output:
            None
    '''
    logging.info("Starting EDA.")

    # Create directories if they do not exist
    os.makedirs('images/EDA', exist_ok=True)

    # Histogram for Churn
    plt.figure(figsize=(20, 10))
    dataframe['Churn'].hist()
    plt.savefig('images/EDA/churn_histogram.png')

    # Histogram for Customer Age
    plt.figure(figsize=(20, 10))
    dataframe['Customer_Age'].hist()
    plt.savefig('images/EDA/customer_age_histogram.png')

    # Bar plot for Marital Status
    plt.figure(figsize=(20, 10))
    dataframe['Marital_Status'].value_counts('normalize').plot(kind='bar')
    plt.savefig('images/EDA/marital_status_bar.png')

    # KDE plot for Total Transaction Count
    plt.figure(figsize=(20, 10))
    sns.histplot(dataframe['Total_Trans_Ct'], kde=True)
    plt.savefig('images/EDA/total_trans_ct_distplot.png')

    # Heatmap for correlations
    plt.figure(figsize=(20, 10))
    sns.heatmap(dataframe.corr(), annot=False, cmap='Dark2_r', linewidths=2)
    plt.savefig('images/EDA/correlation_heatmap.png')

    logging.info("EDA completed.")


def encoder_helper(dataframe, category_lst, response) -> pd.DataFrame:
    '''
    helper function to turn each categorical column into a new column with
    propotion of churn for each category - associated with cell 15 from the notebook

    input:
            dataframe: pandas dataframe
            category_lst: list of columns that contain categorical features
            response: string of response name [optional argument that could
            be used for naming variables or index y column]

    output:
            dataframe: pandas dataframe with new columns for
    '''
    group_means = dataframe.groupby(category_lst).mean()[response]
    dataframe[category_lst + '_Churn'] = dataframe[category_lst].map(group_means)
    return dataframe


def perform_feature_engineering(dataframe, response) -> list:
    '''
    input:
              dataframe: pandas dataframe
              response: string of response name [optional argument
              that could be used for naming variables or index y column]

    output:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    '''
    # List of categorical columns
    cat_columns = ['Gender', 'Education_Level', 'Marital_Status',
                   'Income_Category', 'Card_Category']

    # Encoding categorical columns
    for col in cat_columns:
        dataframe = encoder_helper(dataframe, col, response)

    # List of quantitative columns
    quant_columns = [
        'Customer_Age',
        'Dependent_count',
        'Months_on_book',
        'Total_Relationship_Count',
        'Months_Inactive_12_mon',
        'Contacts_Count_12_mon',
        'Credit_Limit',
        'Total_Revolving_Bal',
        'Avg_Open_To_Buy',
        'Total_Amt_Chng_Q4_Q1',
        'Total_Trans_Amt',
        'Total_Trans_Ct',
        'Total_Ct_Chng_Q4_Q1',
        'Avg_Utilization_Ratio']

    # Append encoded categorical column names
    encoded_cat_columns = [x + '_Churn' for x in cat_columns]
    keep_cols = quant_columns + encoded_cat_columns

    X = dataframe[keep_cols]
    y = dataframe[response]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42)
    return X_train, X_test, y_train, y_test


def classification_report_image(y_train,
                                y_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf) -> None:
    '''
    produces classification report for training and testing results and stores report as image
    in images folder
    input:
            y_train: training response values
            y_test:  test response values
            y_train_preds_lr: training predictions from logistic regression
            y_train_preds_rf: training predictions from random forest
            y_test_preds_lr: test predictions from logistic regression
            y_test_preds_rf: test predictions from random forest

    output:
             None
    '''
    # Create directories if they do not exist
    os.makedirs('images/report', exist_ok=True)

    # Save Random Forest classification report as an image
    plt.figure(figsize=(10, 5))
    plt.text(
        0.01,
        1,
        "Random Forest - Training\n" +
        classification_report(
            y_train,
            y_train_preds_rf),
        fontproperties='monospace',
        verticalalignment='top')
    plt.text(
        0.01,
        0.5,
        "Random Forest - Testing\n" +
        classification_report(
            y_test,
            y_test_preds_rf),
        fontproperties='monospace',
        verticalalignment='top')
    plt.axis('off')
    plt.savefig('images/report/rf_classification_report.png')
    plt.close()

    # Save Logistic Regression classification report as an image
    plt.figure(figsize=(10, 5))
    plt.text(
        0.01,
        1,
        "Logistic Regression - Training\n" +
        classification_report(
            y_train,
            y_train_preds_lr),
        fontproperties='monospace',
        verticalalignment='top')
    plt.text(
        0.01,
        0.5,
        "Logistic Regression - Testing\n" +
        classification_report(
            y_test,
            y_test_preds_lr),
        fontproperties='monospace',
        verticalalignment='top')
    plt.axis('off')
    plt.savefig('images/report/lr_classification_report.png')
    plt.close()


def feature_importance_plot(model, X_data, output_pth) -> None:
    '''
    creates and stores the feature importances in pth
    input:
            model: model object containing feature_importances_
            X_data: pandas dataframe of X values
            output_pth: path to store the figure

    output:
             None
    '''
    # Create directories if they do not exist
    os.makedirs('./importance_features', exist_ok=True)

    # Feature Importance
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    names = [X_data.columns[i] for i in indices]

    # Create figure
    plt.figure(figsize=(20, 5))
    plt.title("Feature Importance")
    plt.ylabel('Importance')
    plt.bar(range(X_data.shape[1]), importances[indices])
    plt.xticks(range(X_data.shape[1]), names, rotation=90)
    plt.savefig(output_pth)


def train_models(X_train, X_test, y_train, y_test) -> None:
    '''
    train, store model results: images + scores, and store models
    input:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    output:
              None
    '''
    logging.info("Starting model training.")

    # Create directories if they do not exist
    os.makedirs('./models', exist_ok=True)
    os.makedirs('images/train', exist_ok=True)

    # Estimators
    rfc = RandomForestClassifier(random_state=42)
    lrc = LogisticRegression(solver='lbfgs', max_iter=3000)

    # Hyperparameters
    param_grid = {
        'n_estimators': [200, 500],
        'max_features': ['auto', 'sqrt'],
        'max_depth': [4, 5, 100],
        'criterion': ['gini', 'entropy']
    }

    # GridSearch
    cv_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)
    cv_rfc.fit(X_train, y_train)

    # Train
    lrc.fit(X_train, y_train)

    y_train_preds_rf = cv_rfc.best_estimator_.predict(X_train)
    y_test_preds_rf = cv_rfc.best_estimator_.predict(X_test)

    y_train_preds_lr = lrc.predict(X_train)
    y_test_preds_lr = lrc.predict(X_test)

    # Save model
    joblib.dump(cv_rfc.best_estimator_, './models/rfc_model.pkl')
    joblib.dump(lrc, './models/logistic_model.pkl')

    logging.info("Random Forest Classifier trained and saved.")
    logging.info("Logistic Regression Classifier trained and saved.")

    # Classification Report
    classification_report_image(
        y_train,
        y_test,
        y_train_preds_lr,
        y_train_preds_rf,
        y_test_preds_lr,
        y_test_preds_rf)

    # ROC Logistic Regression
    lrc_plot = plot_roc_curve(lrc, X_test, y_test)
    plt.savefig('images/train/lr_roc_curve.png')
    plt.clf()

    # ROC LR and RF
    plt.figure(figsize=(15, 8))
    plot_roc_curve(
        cv_rfc.best_estimator_,
        X_test,
        y_test,
        ax=plt.gca(),
        alpha=0.8)
    lrc_plot.plot(ax=plt.gca(), alpha=0.8)
    plt.savefig('images/train/lr_roc_curve_TP_FP.png')
    plt.clf()

    # SHAP analysis
    explainer = shap.TreeExplainer(cv_rfc.best_estimator_)
    shap_values = explainer.shap_values(X_test)
    shap.summary_plot(shap_values, X_test, plot_type="bar")
    plt.savefig('images/train/shap_summary_plot.png')
    plt.clf()

    logging.info("Model training completed.")


if __name__ == "__main__":
    # Import data
    data_frame = import_data("data/bank_data.csv")

    # Perform EDA
    perform_eda(data_frame)

    # Train model
    x_train, x_test, y_train, y_test = perform_feature_engineering(
        data_frame, 'Churn')
    train_models(x_train, x_test, y_train, y_test)

    # Load models
    rfc_model = joblib.load('./models/rfc_model.pkl')

    # Feature Importance
    feature_importance_plot(
        rfc_model,
        x_train,
        './images/importance_features/rfc_model.png')
