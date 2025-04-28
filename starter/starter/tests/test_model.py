from starter.ml.model import compute_model_metrics, inference, train_model
from starter.ml.data import process_data
from sklearn.model_selection import train_test_split
import pickle
import logging
import pytest
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO)


def load_pickle(file_path):
    try:
        with open(file_path, "rb") as f:
            return pickle.load(f)

    except FileNotFoundError as e:
        logging.error(f"Error loading {file_path}: {e}")
        raise FileNotFoundError(f"Error: {file_path} not found!")

    except Exception as e:
        logging.error(f"Error loading {file_path}: {e}")
        raise RuntimeError(f"Unexpected error while loading '{file_path}': {str(e)}")


# Load the model
model = load_pickle(
    "/Users/achillejuniormbogoltouye/Documents/mldevops/starter/model/clf_model.pkl"
)


@pytest.fixture
def load_split_data():
    """
    Load the data for testing.
    """
    df = pd.read_csv(
        "/Users/achillejuniormbogoltouye/Documents/mldevops/starter/data/census.csv"
    ).sample(15_000, random_state=42)

    train, test = train_test_split(df, test_size=0.20, random_state=42, shuffle=True)

    # Define categorical features
    cat_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]

    # Proces the train data with the process_data function.
    X_train, y_train, encoder, lb, scaler = process_data(
        train, categorical_features=cat_features, label="salary", training=True
    )
    X_test, y_test, encoder, lb, scaler = process_data(
        test,
        categorical_features=cat_features,
        label="salary",
        training=False,
        lb=lb,
        encoder=encoder,
        scaler=scaler,
    )
    return X_train, y_train, X_test, y_test


def test_train_model(load_split_data):
    """
    Test the train_model function with a sample input.
    """
    # Sample input data
    X_train, y_train, _, _ = load_split_data

    # Train the model
    model = train_model(X_train, y_train)

    # Check if the model is trained
    assert model is not None
    assert isinstance(model, type(model))

    # Check if the model has the expected attributes
    assert hasattr(model, "predict")


def test_compute_metrics(load_split_data):
    """
    Test the compute_model_metrics function with a sample input.
    """
    # Sample input data
    X_train, y_train, X_test, y_test = load_split_data

    assert X_train.shape[1] == X_test.shape[1]
    assert X_train.shape[0] == y_train.shape[0]
    assert X_test.shape[0] == y_test.shape[0]

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # Check if the predictions are of the expected type
    assert isinstance(y_train_pred, np.ndarray)
    assert isinstance(y_test_pred, np.ndarray)

    precision, recall, fbeta = compute_model_metrics(y_train, y_train_pred)

    test_precision, test_recall, test_fbeta = compute_model_metrics(y_test, y_test_pred)

    # Check if the metrics are of the expected type
    assert isinstance(precision, float)
    assert isinstance(recall, float)
    assert isinstance(fbeta, float)

    assert isinstance(test_precision, float)
    assert isinstance(test_recall, float)
    assert isinstance(test_fbeta, float)

    # Check if the metrics are within expected ranges
    assert 0 <= precision <= 1
    assert 0 <= recall <= 1
    assert 0 <= fbeta <= 1

    assert 0 <= test_precision <= 1
    assert 0 <= test_recall <= 1
    assert 0 <= test_fbeta <= 1


def test_inference(load_split_data):
    """Test the inference function with a sample input."""

    # Sample input data
    X_train, y_train, X_test, y_test = load_split_data

    # Predict
    train_pred = inference(model, X_train)
    test_pred = inference(model, X_test)

    # Check if the predictions are of the expected type
    assert isinstance(train_pred, np.ndarray)
    assert isinstance(test_pred, np.ndarray)

    # Check if the predictions are of the expected shape
    assert train_pred.shape == y_train.shape
    assert test_pred.shape == y_test.shape

    train_pred = train_pred.tolist()
    test_pred = test_pred.tolist()

    assert isinstance(train_pred, list) and all(isinstance(x, int) for x in train_pred)
    assert isinstance(test_pred, list) and all(isinstance(x, int) for x in test_pred)
