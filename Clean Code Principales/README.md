# Predict Customer Churn

Project **Predict Customer Churn** of ML DevOps Engineer Nanodegree Udacity This project focuses on implementing machine learning models to predict customer churn and integrating good software engineering practices.

## Project Description
The goal of this project is to identify customers likely to churn and understand key factors leading to customer churn. We use various machine learning techniques to predict churn and implement software best practices, including logging, testing, and modular coding. The project demonstrates the ability to combine machine learning engineering with solid DevOps processes.

## Files and data description
`churn_library.py`: This Python script contains all the necessary functions to perform data importing, exploratory data analysis (EDA), feature engineering, model training, and evaluation. It utilizes various libraries such as Pandas, NumPy, Scikit-learn, Matplotlib, and Seaborn.

`churn_script_logging_and_testing.py`: This Python script is dedicated to testing the functions defined in churn_library.py. It includes tests for data importing, EDA, feature engineering, and model training. The script also contains logging configurations to record the flow of the script execution and catch any potential issues.

`data/`: This directory contains the dataset used for training and testing the model. It typically consists of customer data, including both features and a target variable indicating churn.

`models/`: This directory is used to store the trained machine learning models, which are saved in a serialized format (e.g., .pkl files).

`images/EDA`: This directory stores the images/plots generated during the exploratory data analysis (EDA) phase.

`images/reports`: This directory stores the images/plots generated during the classification models.

`images/reports`: This directory stores the images/plots generated during the training.

## Running Files
To execute the scripts successfully, follow these steps:

1. Data Preparation: Ensure that the dataset is placed in the data/ directory.

2. Run Churn Library Script: Execute `churn_library.py` to perform data preprocessing, EDA, feature engineering, model training, and evaluation. This script will generate models in the models/ directory and EDA plots in the images/ directory.

```
python churn_library.py
```

3. Run Testing and Logging Script: Execute `churn_script_logging_and_testing.py` to run tests on the functions defined in `churn_library.py`. This will also generate log files that record the process and outcomes of the tests.


```
python churn_script_logging_and_testing.py
```

When you run these files, you should see logs being generated that inform you of the process, any errors or issues encountered, and the success of different operations. The tests will validate the correctness of the functions and the integrity of the data processing and model training.
