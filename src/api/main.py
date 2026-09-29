from pathlib import Path

import joblib
from fastapi import FastAPI

from src.api.inference import create_features


app = FastAPI(
    title="CBDC Fraud Detection API",
    description="Fraud detection using Random Forest and XGBoost",
    version="1.0.0",
)


BASE_DIR = Path(__file__).resolve().parents[2]

rf_model = joblib.load(
    BASE_DIR / "models" / "random_forest.joblib"
)

xgb_model = joblib.load(
    BASE_DIR / "models" / "xgboost.joblib"
)


@app.get("/")
def home():
    return {
        "message": "CBDC Fraud Detection API is running"
    }


@app.post("/predict")
def predict(transaction: dict):
    features = create_features(transaction)

    rf_probability = rf_model.predict_proba(features)[0][1]
    xgb_probability = xgb_model.predict_proba(features)[0][1]

    return {
        "random_forest": {
            "prediction": "fraud" if rf_probability >= 0.5 else "normal",
            "fraud_probability": round(float(rf_probability), 4),
        },
        "xgboost": {
            "prediction": "fraud" if xgb_probability >= 0.5 else "normal",
            "fraud_probability": round(float(xgb_probability), 4),
        },
    }