import pandas as pd
import numpy as np


numeric_features = [
    "amount_eur",
    "transactions_1h",
    "transactions_24h",
    "avg_amount_7d",
    "amount_deviation",
    "abs_amount_deviation",
    "sender_balance",
    "receiver_balance",
    "sender_velocity",
    "receiver_velocity",
    "amount_to_avg_7d",
    "amount_to_sender_balance",
    "is_international",
    "new_device",
    "new_location",
    "hour",
    "day_of_week",
    "is_weekend",
]

categorical_features = [
    "transaction_type",
    "payment_mode",
    "country",
    "merchant_category",
]


def create_features(transaction):
    df = pd.DataFrame([transaction])

    df["timestamp"] = pd.to_datetime(df["timestamp"])

    df["hour"] = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

    df["amount_to_avg_7d"] = (
        df["amount_eur"] /
        df["avg_amount_7d"].replace(0, np.nan)
    )

    df["amount_to_avg_7d"] = (
        df["amount_to_avg_7d"]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(1.0)
    )

    df["amount_to_sender_balance"] = (
        df["amount_eur"] /
        df["sender_balance"].replace(0, np.nan)
    )

    df["amount_to_sender_balance"] = (
        df["amount_to_sender_balance"]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0)
    )

    df["abs_amount_deviation"] = df["amount_deviation"].abs()

    feature_columns = numeric_features + categorical_features

    return df[feature_columns]