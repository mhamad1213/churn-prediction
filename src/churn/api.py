"""Milestone 6: serve the churn model as a web API.

Run locally:   uvicorn churn.api:app --reload --app-dir src
Then open:     http://127.0.0.1:8000        (the web page)
               http://127.0.0.1:8000/docs   (interactive API docs)

By default the model comes from models/best_model.joblib. Set MODEL_URI to load
a version from the MLflow registry instead, e.g. MODEL_URI=models:/churn-model@champion
"""

import json
import os
from pathlib import Path
from typing import Literal

import joblib
import pandas as pd
from fastapi import BackgroundTasks, FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from churn.config import MODELS_DIR
from churn.data import input_schema
from churn.explain import Explainer
from churn.prediction_log import log_prediction, summary

YesNo = Literal["Yes", "No"]
AddOn = Literal["Yes", "No", "No internet service"]


class Customer(BaseModel):
    """One customer's details. Pydantic rejects anything that doesn't fit."""

    gender: Literal["Male", "Female"]
    SeniorCitizen: YesNo
    Partner: YesNo
    Dependents: YesNo
    tenure: int = Field(ge=0, le=100, description="Months as a customer")
    PhoneService: YesNo
    MultipleLines: Literal["Yes", "No", "No phone service"]
    InternetService: Literal["DSL", "Fiber optic", "No"]
    OnlineSecurity: AddOn
    OnlineBackup: AddOn
    DeviceProtection: AddOn
    TechSupport: AddOn
    StreamingTV: AddOn
    StreamingMovies: AddOn
    Contract: Literal["Month-to-month", "One year", "Two year"]
    PaperlessBilling: YesNo
    PaymentMethod: Literal[
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ]
    MonthlyCharges: float = Field(gt=0)
    TotalCharges: float = Field(ge=0)

    model_config = {
        "json_schema_extra": {
            "example": {
                "gender": "Female", "SeniorCitizen": "No", "Partner": "No",
                "Dependents": "No", "tenure": 3, "PhoneService": "Yes",
                "MultipleLines": "No", "InternetService": "Fiber optic",
                "OnlineSecurity": "No", "OnlineBackup": "No", "DeviceProtection": "No",
                "TechSupport": "No", "StreamingTV": "Yes", "StreamingMovies": "No",
                "Contract": "Month-to-month", "PaperlessBilling": "Yes",
                "PaymentMethod": "Electronic check", "MonthlyCharges": 85.0,
                "TotalCharges": 255.0,
            }
        }
    }


class Reason(BaseModel):
    feature: str
    value: str | int | float
    effect: float


class Prediction(BaseModel):
    churn_probability: float
    contact_customer: bool
    risk_level: Literal["low", "medium", "high"]
    threshold: float
    top_reasons: list[Reason]


def load_model():
    uri = os.environ.get("MODEL_URI")
    if not uri:
        return joblib.load(MODELS_DIR / "best_model.joblib"), "models/best_model.joblib"
    # Only this path needs MLflow, so the Docker image can stay without it
    import mlflow

    from churn.track import tracking_uri

    mlflow.set_tracking_uri(tracking_uri())
    return mlflow.sklearn.load_model(uri), uri


# Loaded once when the server starts, not on every request
model, model_source = load_model()
threshold = json.loads((MODELS_DIR / "threshold.json").read_text())["threshold"]
explainer = Explainer(model)

app = FastAPI(title="Customer Churn Predictor", version="1.0")
STATIC = Path(__file__).parent / "static"


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/model-info")
def model_info() -> dict:
    return {"source": model_source, "threshold": threshold}


@app.get("/monitoring/summary")
def monitoring_summary() -> dict:
    return {"model": model_source, **summary()}


@app.get("/", include_in_schema=False)
def home() -> FileResponse:
    return FileResponse(STATIC / "index.html")


@app.post("/predict", response_model=Prediction)
def predict(customer: Customer, background_tasks: BackgroundTasks) -> Prediction:
    df = input_schema.validate(pd.DataFrame([customer.model_dump()]))
    prob = float(model.predict_proba(df)[0, 1])
    risk = "high" if prob >= 0.7 else "medium" if prob >= threshold else "low"
    # saved after the response goes out, so logging never slows the user down
    background_tasks.add_task(log_prediction, customer.model_dump(), prob, prob >= threshold, model_source)
    return Prediction(
        churn_probability=round(prob, 3),
        contact_customer=prob >= threshold,
        risk_level=risk,
        threshold=threshold,
        top_reasons=explainer.top_reasons(df),
    )
