from pathlib import Path
from setuptools import find_namespace_packages, setup

# Load packages from requirements.txt
BASE_DIR = Path(__file__).parent
with open(Path(BASE_DIR, "requirements.txt"), "r") as file:
    required_packages = [ln.strip() for ln in file.readlines()]

setuptools.setup(
    name="ML model to cloud",
    version="0.0.0",
    description="Deploying a ML model to cloud application platform\
        with FastAPI",
    author="Ricard Santiago Raigada García",
    packages=find_namespace_packages(),
    install_requires=[required_packages],
)
