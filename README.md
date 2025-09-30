# Scoring service

![Request form](docs/form.png)

# Credit Scoring

A small credit scoring project containing data, training code and a minimal inference service.

Main goal is to make predictions on minimal data listed on screenshot.

**Model and Performance:**

* Best Model: XGBoost

* Test Recall: 0.6933

* Test Precision: 0.1398

* Denied Rate: 57%

Notes on Performance:
The metrics indicate that the model prioritizes recall, meaning it identifies a high proportion of actual positives (e.g., risky credit applications) but at the cost of low precision, resulting in many false positives. These results are expected given the minimal size of the dataset and limited feature set, which constrain the model’s predictive capability. Despite these limitations, the project demonstrates an end-to-end workflow from data ingestion to inference, providing a foundation for further improvements with more data.

Repository includes scripts to train a model, a saved model artifact, and a lightweight service to score new examples.

## Quick Start

To copy bad run project follow these instructions

```powershell
git clone https://github.com/rimeger/credit_scoring

docker-compose up --build
```

This will:

- build images from `service/Dockerfile` and `app/Dockerfile`
- start the FastAPI service on host port 7000
- start the Streamlit UI on host port 7001

Access the running services in your browser:

- FastAPI Swagger UI: http://localhost:7000/docs
- Streamlit UI: http://localhost:7001

## Training

To retrain the model from `data/scoring.csv` run:

```powershell
python train.py
```

When training completes it should write a model artifact (e.g. `models/best_model.pkl`).


