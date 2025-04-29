import pandas as pd

# import requests
# import json
import logging
import os

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# retrieve the base directory
base_dir = os.path.dirname(__file__)

# load the data
df = pd.read_csv(os.path.join(base_dir, "data", "census.csv"))

df.columns = df.columns.str.strip()
df = df.drop(columns=["salary"])


# sample data convert to dict
data = df.iloc[1].to_dict()
logger.info(f"Data to be sent: {data}")


# Define the endpoint URL
url = "http://127.0.0.1:8000/predict"

# Send a POST request to the API
# response = requests.post(url, data=json.dumps(data))


# Check the response status code
# if response.status_code == 200:
# logger.info("Prediction successful!")
# prediction = response.json()
# print((prediction))
