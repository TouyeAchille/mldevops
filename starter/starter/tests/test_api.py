from fastapi.testclient import TestClient
from main import app

client = TestClient(app)


def test_root():
    """
    Test the root endpoint to ensure it redirects to the docs.
    """
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {
        "message": "Welcome to the Census Income Prediction API!"
    }


def test_predict():
    """
    Test the predict endpoint with valid data.
    """
    # Sample input data
    response = client.post(
        "/predict",
        json={
            "age": 50,
            "workclass": " Self-emp-not-inc",
            "fnlgt": 83311,
            "education": " Bachelors",
            "education-num": 13,
            "marital-status": " Married-civ-spouse",
            "occupation": " Exec-managerial",
            "relationship": " Husband",
            "race": " White",
            "sex": " Male",
            "capital-gain": 0,
            "capital-loss": 0,
            "hours-per-week": 13,
            "native-country": " United-States",
        },
    )
    assert response.status_code == 200
    assert response.json() == {
        "salary_prediction": [0],
        "salary_prediction_label": [" <=50K"],
    }


def test_predict_result():
    """
    Test the predict endpoint with invalid data.
    """
    # Sample input data
    response = client.post(
        "/predict",
        json={
            "age": 50,
            "workclass": " Self-emp-not-inc",
            "fnlgt": 83311,
            "education": " Bachelors",
            "education-num": 13,
            "marital-status": " Married-civ-spouse",
            "occupation": " Exec-managerial",
            "relationship": " Husband",
            "race": " White",
            "sex": " Male",
            "capital-gain": 0,
            "capital-loss": 0,
            "hours-per-week": 13,
            "native-country": " United-States",
        },
    )

    # Check the response
    assert response.status_code == 200
    assert response.json() == {
        "salary_prediction": [0],
        "salary_prediction_label": [" <=50K"],
    }
    assert isinstance(response.json()["salary_prediction"], list) and all(
        isinstance(x, int) for x in response.json()["salary_prediction"]
    )
