# Put the code for your API here.
import pickle
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from starter.ml.data import process_data
from starter.ml.model import compute_model_metrics, inference


# data model
class Input_Features(BaseModel):
    # Define the input features with their types
    workclass: str
    education: str
    marital_status: str = Field(..., alias="marital-status")
    occupation: str
    relationship: str
    race: str
    sex: str
    age: int
    capital_gain: int = Field(..., alias="capital-gain")
    capital_loss: int = Field(..., alias="capital-loss")
    hours_per_week: int = Field(..., alias="hours-per-week")
    native_country: str = Field(..., alias="native-country")
    education_num: int = Field(..., alias="education-num")
    fnlwgt: int

    class Config:
        populate_by_name = True  # Allows aliasing (useful for column names with dashes)


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


# Chargement des modèles avec gestion des erreurs
def load_pickle(file_path):
    try:
        with open(file_path, "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        raise HTTPException(status_code=500, detail=f"Error: {file_path} not found!")


model = load_pickle(
    "/Users/achillejuniormbogoltouye/Documents/mldevops/starter/model/clf_model.pkl"
)
encoder = load_pickle(
    "/Users/achillejuniormbogoltouye/Documents/mldevops/starter/model/encoder.pkl"
)
lb = load_pickle(
    "/Users/achillejuniormbogoltouye/Documents/mldevops/starter/model/lb.pkl"
)
scaler = load_pickle(
    "/Users/achillejuniormbogoltouye/Documents/mldevops/starter/model/scaler.pkl"
)

# create the FastAPI app
app = FastAPI()

# Define the root endpoint
# This is the main entry point of the API
# It returns a welcome message
# when the root URL ("/") is accessed


@app.get("/")
async def root():
    return {
        "message": """Welcome to machine learning devops project: The goal is to develop a classification model on publicly available Census Bureau data and
                          create unit tests to monitor the model performance on various data slices. Then, deploy model using the FastAPI package and create API tests.
                          The slice validation and the API tests will be incorporated into a CI/CD framework using GitHub Actions
            """
    }


# Define the inference endpoint
# This endpoint is used to make predictions using the trained model
# It accepts a POST request with input features in JSON format
# The input features are validated using the Input_Features model


@app.post("/predict")
async def predict(input_features: Input_Features):
    """
    Predict the salary based on input features.
    """
    try:
        # Convert data model to  dataframe
        input_dict = input_features.model_dump(by_alias=True)
        # input_df = pd.DataFrame([input_dict])

        # process input data
        # X, _, _, _ = process_data(
        # input_df, categorical_features=cat_features, label="salary",
        # training=False, encoder=encoder, lb=lb, scaler=scaler
        # )

        # predict
        # prediction = inference(model, X)

        return input_features  # prediction.tolist()}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
