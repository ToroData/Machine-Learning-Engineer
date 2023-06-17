# Machine Learning Engineer and Deep Learning Projects

<div align="center">
    <img src="https://thedatascientist.digital/img/logo.png" alt="Logo" width="25%">
</div>


This repository contains several Machine Learning and Deep Learning projects, including neural network training, image classification using AWS SageMaker, and an ML workflow based on Step Functions. The projects are designed to demonstrate technical skills in areas such as data manipulation, data modeling, and model deployment.

## Technical Skills

The following technical skills have been developed through the projects in this repository:

- Python
- TensorFlow
- PyTorch
- AWS SageMaker
- Step Functions
- Data preprocessing and manipulation
- Model selection and evaluation
- Model deployment
## Projects

### Deep Learning

-  training_a_cnn.ipynb: This notebook demonstrates training a Convolutional Neural Network (CNN) on the CIFAR-10 dataset using TensorFlow. The notebook includes data preprocessing, model selection and evaluation, and model visualization.

### Image Classification using AWS SageMaker
- train_and_deploy.ipynb: This notebook demonstrates training and deploying a custom image classification model using AWS SageMaker. The notebook includes data preprocessing, model selection and evaluation, hyperparameter tuning using Amazon SageMaker Automatic Model Tuning, and model deployment to a SageMaker endpoint.

### ML Workflow
This project includes a workflow for classifying images using a Step Functions state machine. The following components are included:

`lambdas`: AWS Lambda functions used by the state machine.

`classifyImage.py`: Classifies an image using a pre-trained model.

`confidenceImage.py`: Calculates the confidence level of the classification result.

`serializeImageData.py`: Serializes the image data for use in the state machine.

`notebooks`: Jupyter Notebook used to demonstrate the workflow.

`Scones_Unlimited.ipynb`: This notebook demonstrates the ML workflow by classifying images of scones.

`stepFunctions`: JSON file defining the state machine and its transitions.

`img`: Sample images used in the workflow.

## File Structure


```
│   LICENSE
│   README.md
│
├───Deep Learning
│       training_a_cnn.ipynb
│
├───Image Classification using AWS SageMaker
│   │   hpo.py
│   │   LICENSE.txt
│   │   README.md
│   │   train_and_deploy.ipynb
│   │   train_model.py
│   │
│   └───unzipped
│           README.md
│           requirements.txt
│           zip.py
│
└───ML-Workflow
    │   README.md
    │
    ├───lambdas
    │       classifyImage.py
    │       confidenceImage.py
    │       serializeImageData.py
    │
    ├───notebooks
    │       Scones_Unlimited.ipynb
    │
    └───stepFunctions
        │   stepFunction.json
        │
        └───img
                all_states_green.png
                two_states_green_95_conf.png
```
## Learning

These projects have been developed as part of the Machine Learning Engineer Nanodegree program at Udacity. Through the program, I have gained hands-on experience in developing and deploying machine learning models using a variety of tools and techniques.
## Replication

If you would like to replicate any of the projects in this repository, please refer to the README file in each project folder for instructions.
## License
This project is licensed under the MIT License - see the [MIT LICENSE](https://choosealicense.com/licenses/mit/) file for details.


## Author

- [@RicardSantiagoRaigadaGarcía](https://www.thedatascientist.digital/)