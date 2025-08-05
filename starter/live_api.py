import argparse
import json
import logging
import os

import pandas as pd
import requests

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

parser = argparse.ArgumentParser()
parser.add_argument("--host", default="127.0.0.1")
parser.add_argument("--port", default=8000, type=int)
args = parser.parse_args()

# Define the endpoint URL
url = f"http://{args.host}:{args.port}/predict"

# Send a POST request to the API
response = requests.post(url, data=json.dumps(data))


# Check the response status code
print(response.status_code == 200)
logger.info("Prediction successful!")

prediction = response.json()
print((prediction))
