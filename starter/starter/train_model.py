# Script to train machine learning model.
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from ml.data import process_data
from ml.model import compute_model_metrics, train_model, inference


# load in the data
df = pd.read_csv(
    "/Users/achillejuniormbogoltouye/Documents/mldevops/starter/data/census.csv"
)

# Optional enhancement, use K-fold cross validation instead of a train-test split.
train, test = train_test_split(df, test_size=0.20, random_state=42, shuffle=True)

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
X_test, y_test, encoder, lb, scaler = process_data(
    test,
    categorical_features=cat_features,
    label="salary",
    training=False,
    lb=lb,
    encoder=encoder,
    scaler=scaler,
)


# Train model
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


def save_pickle(file_path):
    with open(file_path, "wb") as file:
        pickle.dump(clf_model, file)


# save model
filename = (
    "/Users/achillejuniormbogoltouye/Documents/mldevops/starter/model/clf_model.pkl"
)
save_pickle(filename)
print()
print("=" * 10)
print(f"Model saved at {filename}")

# save encoder, lb and scaler
filename = (
    "/Users/achillejuniormbogoltouye/Documents/mldevops/starter/model/encoder.pkl"
)
save_pickle(filename)
print("=" * 10)
print(f"Encoder saved at {filename}")


filename = "/Users/achillejuniormbogoltouye/Documents/mldevops/starter/model/lb.pkl"
save_pickle(filename)
print("=" * 10)
print(f"lb saved at {filename}")


filename = "/Users/achillejuniormbogoltouye/Documents/mldevops/starter/model/scaler.pkl"
save_pickle(filename)
print("=" * 10)
print(f"scaler saved at {filename}")
