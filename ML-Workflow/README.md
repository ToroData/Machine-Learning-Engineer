# Scone Unlimited

We'll use a sample dataset called CIFAR to simulate the challenges Scones Unlimited are facing in Image Classification. In order to start working with CIFAR we'll need to:

1. Extract the data from a hosting service
2. Transform it into a usable shape and format
3. Load it into a production system

In other words, we're going to do some simple ETL!

## Files structure
```bash
.
├── lambdas/
│   ├── classifyImage.py
│   ├── confidenceImage.py
│   └── serializeImage.py
├── notebooks/
│   └── Scones_Unlimited.ipynb
├── stepFunctions/
│   ├── stepFunction.json
│   └── img/
│       ├── all_states_green.png
│       └── two_states_green_95_conf.png
└── README.md
```

## Features

- Step Functions
- Lambda
- S3

## Installation

Install this project by running:

## Usage
Open your AWS console and open Sagemaker if you want to run the code in the cloud.

Notes about the instance size and kernel setup: this notebook has been tested on

1. The `Python 3 (Data Science)` kernel
2. The `ml.t3.medium` Sagemaker notebook instance


## Contributing
- Fork the repository.
- Create a new branch for your changes.
- Make your changes and commit them with descriptive messages.
- Push your changes to your forked repository.
- Open a pull request with a detailed explanation of your changes.

## License
This project is licensed under the  License.