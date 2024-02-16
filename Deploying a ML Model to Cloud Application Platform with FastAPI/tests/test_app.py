from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_read_main():
    """
    Test para el endpoint GET en la raíz.
    """
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Welcome to the Census Income Prediction API!"}


def test_post_inference_less_than_50k():
    """
    Test para el endpoint POST que debería predecir ingresos <=50K.
    """
    data = {
        "age": 39,
        "work-class": "State-gov",
        "fnlgt": 77516,
        "education": "Bachelors",
        "education-num": 13,
        "marital-status": "Never-married",
        "occupation": "Adm-clerical",
        "relationship": "Not-in-family",
        "race": "White",
        "sex": "Male",
        "capital-gain": 2174,
        "capital-loss": 0,
        "hours-per-week": 40,
        "native-country": "United-States"
    } # line 1 census.csv
    response = client.post("/inference/", json=data)
    assert response.status_code == 200
    assert response.json()["prediction"] == "<=50K"


def test_post_inference_greater_than_50k():
    """
    Test para el endpoint POST que debería predecir ingresos >50K.
    """
    data = {
        "age": 52,
        "work-class": "Self-emp-not-inc",
        "fnlgt": 209642,
        "education": "HS-grad",
        "education-num": 9,
        "marital-status": "Married-civ-spouse",
        "occupation": "Exec-managerial",
        "relationship": "Husband",
        "race": "White",
        "sex": "Male",
        "capital-gain": 0,
        "capital-loss": 0,
        "hours-per-week": 45,
        "native-country": "United-States"
    } # line 9 census.csv
    response = client.post("/inference/", json=data)
    assert response.status_code == 200
    assert response.json()["prediction"] == ">50K"
