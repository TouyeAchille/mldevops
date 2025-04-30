# Script to train machine learning model.
import pandas as pd
import os
import pickle
from sklearn.model_selection import train_test_split
from starter.ml.data import process_data
from starter.ml.model import compute_model_metrics, train_model, inference
from pathlib import Path

current_dir = Path(__file__).resolve().parent
# load in the data
df = pd.read_csv(os.path.join(current_dir.parent, "data", "census.csv"))
# Optional enhancement, use K-fold cross validation instead of a train-test split.
train, test = train_test_split(df, test_size=0.20, random_state=42, shuffle=True)

# Define the categorical features
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

# Proces the test data with the process_data function.
X_test, y_test, encoders, lbs, scalers = process_data(
    test,
    categorical_features=cat_features,
    label="salary",
    training=False,
    lb=lb,
    encoder=encoder,
    scaler=scaler,
)


# Train mod
print("Training model...")
clf_model = train_model(X_train, y_train)
print("Training model finish...")
print("=" * 30)
print()

# test prediction
y_test_pred = inference(clf_model, X_test)

# train prediction
y_train_pred = inference(clf_model, X_train)

# evaluate the model
print("Evaluating model on train data...")
precision1, recall1, fbeta1 = compute_model_metrics(y_train, y_train_pred)

print(f"Precision: {precision1}")
print(f"Recall: {recall1}")
print(f"F1: {fbeta1}")
print("=" * 10)
print()

# evaluate the model
print("Evaluating model on test data...")
precision2, recall2, fbeta2 = compute_model_metrics(y_test, y_test_pred)

print(f"Precision: {precision2}")
print(f"Recall: {recall2}")
print(f"F1: {fbeta2}")


def save_pickle(file_path, objet_to_save):
    """
    Save the object to a pickle file."""
    with open(file_path, "wb") as file:
        pickle.dump(objet_to_save, file)


# save model
path_to_save_model = os.path.join(current_dir.parent, "model", "model.pkl")
save_pickle(path_to_save_model, clf_model)
print()
print("=" * 10)
print(f"Model saved at {path_to_save_model}")

# save encoder
path_to_save_encoder = os.path.join(current_dir.parent, "model", "encoder.pkl")
save_pickle(path_to_save_encoder, encoder)
print("=" * 10)
print(f"Encoder saved at {path_to_save_encoder}")

# save label binarizer
path_to_save_lb = os.path.join(current_dir.parent, "model", "lb.pkl")
save_pickle(path_to_save_lb, lb)
print("=" * 10)
print(f"lb saved at {path_to_save_lb}")

# save scaler
path_to_save_scaler = os.path.join(current_dir.parent, "model", "scaler.pkl")
save_pickle(path_to_save_scaler, scaler)
print("=" * 10)
print(f"scaler saved at {path_to_save_scaler}")
print("=" * 10)
print("Model training and saving completed.")
