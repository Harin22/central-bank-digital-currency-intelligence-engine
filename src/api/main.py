from pathlib import Path

import joblib
from fastapi import FastAPI
from pydantic import BaseModel

from src.api.inference import create_features
from src.api.gnn_inference import predict_gnn


app = FastAPI(
    title="CBDC Fraud Detection API",
    description="Fraud detection using Random Forest, XGBoost and GraphSAGE",
    version="1.0.0",
)


BASE_DIR = Path(__file__).resolve().parents[2]


rf_model = joblib.load(
    BASE_DIR / "models" / "random_forest.joblib"
)

xgb_model = joblib.load(
    BASE_DIR / "models" / "xgboost.joblib"
)


class Transaction(BaseModel):

    sender_id: str
    receiver_id: str

    amount_eur: float

    transactions_1h: int
    transactions_24h: int

    sender_velocity: float
    receiver_velocity: float

    timestamp: str
    avg_amount_7d: float
    amount_deviation: float

    sender_balance: float
    receiver_balance: float | None = None

    is_international: int
    new_device: int
    new_location: int

    transaction_type: str
    payment_mode: str
    country: str
    merchant_category: str


@app.get("/")
def home():

    return {
        "message": "CBDC Fraud Detection API is running"
    }


@app.post("/predict")
def predict(transaction: Transaction):

    transaction_data = transaction.model_dump()


    # RF and XGBoost
    features = create_features(
        transaction_data
    )


    rf_probability = (
        rf_model
        .predict_proba(features)[0][1]
    )


    xgb_probability = (
        xgb_model
        .predict_proba(features)[0][1]
    )


    # GNN
    gnn_result = predict_gnn(
        transaction_data
    )


    return {

        "random_forest": {
            "prediction":
                "fraud"
                if rf_probability >= 0.5
                else "normal",

            "fraud_probability":
                round(
                    float(rf_probability),
                    4,
                ),
        },

        "xgboost": {
            "prediction":
                "fraud"
                if xgb_probability >= 0.5
                else "normal",

            "fraud_probability":
                round(
                    float(xgb_probability),
                    4,
                ),
        },

        "gnn": gnn_result,
    }