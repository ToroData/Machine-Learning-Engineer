
<img src="https://thedatascientist.digital/img/logo.png" alt="Logo" width="25%">




# Unzipper and uploader to S3

This repository contains a Python script that extracts the contents of a ZIP file to a temporary folder and uploads the files to an Amazon S3 bucket.

## Requirements

The projects are developed using Python 3.9 and require the following libraries:

- boto3
- tqdm
- zipfile
- os
- sys
- shutil


These libraries can be installed using pip (help: installation). 

## Installation

1. Install the required Python libraries using pip install -r requirements.txt.
2. Set up your AWS credentials by following the AWS documentation or exectue into SageMaker Studio.
3. Ensure that you have a ZIP file that you want to upload to S3.

```python
  cd unzipper
```
With requirements:
```python
  pip install -r requirements.txt
```

## Usage/Examples

```{python}
python zip.py dogImages.zip dog-image-classifier-project-08052023

```


## Author

- [@RicardSantiagoRaigadaGarcía](https://www.thedatascientist.digital/)



