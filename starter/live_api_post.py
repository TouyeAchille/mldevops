import json
import logging
import os

import pandas as pd
import requests

# Set up logging
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
print()
print("==" * 20)
logger.info(f"Data to be sent: {data}")

# Define the endpoint URL
url = "https://appmls-974dd7e75330.herokuapp.com/predict"

print()
print("==" * 20)
print("url:", url)

# Send a POST request to the API
response = requests.post(url, data=json.dumps(data))
print()
print("==" * 20)
# Check the response status code
print("status_code :", response.status_code == 200)
print("==" * 20)
print()
logger.info("Prediction successful!")

print("==" * 20)
prediction = response.json()
print((prediction))
print("==" * 20)
print()
