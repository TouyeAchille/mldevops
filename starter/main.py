# Put the code for your API here.
import pickle
import pandas as pd
from pathlib import Path
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from starter.ml.data import process_data
from starter.ml.model import inference

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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
    native_country: str = Field(..., alias="native-country")
    age: int
    fnlgt: int
    education_num: int = Field(..., alias="education-num")
    capital_gain: int = Field(..., alias="capital-gain")
    capital_loss: int = Field(..., alias="capital-loss")
    hours_per_week: int = Field(..., alias="hours-per-week")

    # examples of the data app can receive.
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
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
                }
            ]
        }
    }


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

# retrieve the base directory
base_dir = Path(__file__).resolve().parent


# Chargement des modèles avec gestion des erreurs
def load_pickle(file_path):
    try:
        with open(file_path, "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        raise HTTPException(status_code=500, detail=f"Error: {file_path} not found!")


def load_all_models(model_dir: Path):
    return {
        "model": load_pickle(model_dir / "clf_model.pkl"),
        "encoder": load_pickle(model_dir / "encoder.pkl"),
        "lb": load_pickle(model_dir / "lb.pkl"),
        "scaler": load_pickle(model_dir / "scaler.pkl"),
    }


models = load_all_models(base_dir / "model")
model, encoder, lb, scaler = models.values()


# create the FastAPI app
app = FastAPI(
    title="Census Income Prediction API",
    description="API for predicting income based on census data.",
    version="1.0.0",
)


@app.get("/")
async def root():
    return {"message": "Welcome to the Census Income Prediction API!"}


@app.post("/predict")
async def predict(input_features: Input_Features):
    """
    Predict the salary based on input features.
    """
    try:

        logger.info("Received input: %s", input_features.model_dump())
        # Convert Pydantic model to dataframe
        input_dict = input_features.model_dump(by_alias=True)
        df = pd.DataFrame([input_dict])

        # Preprocess input data
        X, _y, _encoder, lbz, _scaler = process_data(
            df,
            categorical_features=cat_features,
            label=None,
            training=False,
            encoder=encoder,
            lb=lb,
            scaler=scaler,
        )

        # Predict
        prediction = inference(model, X)

        return {
            "salary_prediction": prediction.tolist(),
            "salary_prediction_label": lb.inverse_transform(prediction).tolist(),
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
