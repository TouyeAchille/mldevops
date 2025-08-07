# Census Income Classification API - MLDevOps Project

In this project, we develop a **classification model** based on **publicly available Census Bureau data**. The pipeline includes:

* Building and evaluating a classification model
* Writing unit tests to monitor performance across **data slices**
* Serving the model using **FastAPI**
* Creating and running **API tests**
* Automating slice validation and API tests within a **CI/CD workflow** using **GitHub Actions**

For details about the model training and evaluation, refer to the [`model_card.md`](starter/model_card.md) file.

---

##  Running the Project Locally

Follow these steps to run the project on your local machine:

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/mldevops.git
cd mldevops
```

### 2. Create a Virtual Environment (with Conda)

```bash
conda env create -f starter/conda.yaml
```

### 3. Activate the Virtual Environment

```bash
conda activate dev_env
```

### 4. Run the FastAPI Application Locally

```bash
cd starter
uvicorn main:app --host=0.0.0.0 --port=YOUR_PORT
```

Replace `YOUR_PORT` with the desired port number (e.g., `8000`).

---

## Sending Requests to the API

Once the FastAPI server is running, you can send requests to the endpoints.
Refer to the [`live_api.py`](starter/live_api.py) script for example API requests.

---
