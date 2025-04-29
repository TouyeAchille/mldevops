import pickle
import pandas as pd
from ml.data import process_data
from ml.model import compute_model_metrics, inference
import logging
import os
from pathlib import Path

# Set up logging
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


# retrieve the base directory
base_dir = Path(__file__).resolve().parent

# load the data
df = pd.read_csv(os.path.join(base_dir.parent, "data", "census.csv"))
df.columns = df.columns.str.strip()


# load the model
model = load_pickle(os.path.join(base_dir.parent, "model", "clf_model.pkl"))

# load the encoder
encoder = load_pickle(os.path.join(base_dir.parent, "model", "encoder.pkl"))

# load the label binarizer and scaler
lb = load_pickle(os.path.join(base_dir.parent, "model", "lb.pkl"))

# load the scaler
scaler = load_pickle(os.path.join(base_dir.parent, "model", "scaler.pkl"))


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


def compute_metrics_on_slices(slicing_feature):
    """
    Compute model performance metrics on slices
    of the data based on a specific feature.
    """

    slices_results = {}

    df = pd.read_csv(os.path.join(base_dir.parent, "data", "census.csv"))
    df.columns = df.columns.str.strip()

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

    unique_values = df[slicing_feature].unique().tolist()
    logging.info(f"Unique values for {slicing_feature}: {unique_values}")

    for value in unique_values:
        logging.info(f"Processing value: {value}")
        slice_idx = df[slicing_feature] == value
        X_slice = df[slice_idx].drop(columns=["salary"])
        y_slice_label = lb.transform(df[slice_idx]["salary"].values).ravel()

        X_slice_test, _, _, _, _ = process_data(
            X_slice,
            categorical_features=cat_features,
            label=None,
            training=False,
            lb=lb,
            encoder=encoder,
            scaler=scaler,
        )

        pred = inference(model, X_slice_test)
        precision, recall, fbeta = compute_model_metrics(y_slice_label, pred)
        logging.info(f"Metrics for {slicing_feature} = {value}:")

        slices_results[value] = {
            "precision": precision,
            "recall": recall,
            "fbeta": fbeta,
        }

    pd.DataFrame.from_dict(slices_results, orient="index").to_csv(
        os.path.join(base_dir, "slices_output_metrics.csv")
    )

    return precision, recall, fbeta


if __name__ == "__main__":
    compute_metrics_on_slices("marital-status")
