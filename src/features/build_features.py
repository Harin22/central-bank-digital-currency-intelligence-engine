import pandas as pd
import numpy as np
from pathlib import Path


INPUT_PATH = Path("data/processed/transactions_with_fraud.csv")
OUTPUT_PATH = Path("data/processed/ml_features.csv")


# load data
df = pd.read_csv(INPUT_PATH)

df["timestamp"] = pd.to_datetime(df["timestamp"])


# timestamp features
df["hour"] = df["timestamp"].dt.hour
df["day_of_week"] = df["timestamp"].dt.dayofweek
df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)


# amount compared to 7-day average
df["amount_to_avg_7d"] = (
    df["amount_eur"] /
    df["avg_amount_7d"].replace(0, np.nan)
)

df["amount_to_avg_7d"] = (
    df["amount_to_avg_7d"]
    .replace([np.inf, -np.inf], np.nan)
    .fillna(1.0)
)


# amount compared to sender balance
df["amount_to_sender_balance"] = (
    df["amount_eur"] /
    df["sender_balance"].replace(0, np.nan)
)

df["amount_to_sender_balance"] = (
    df["amount_to_sender_balance"]
    .replace([np.inf, -np.inf], np.nan)
    .fillna(0)
)


# absolute amount deviation
df["abs_amount_deviation"] = df["amount_deviation"].abs()


# numerical features
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


# categorical features
categorical_features = [
    "transaction_type",
    "payment_mode",
    "country",
    "merchant_category",
]


# create feature dataset
feature_columns = numeric_features + categorical_features

features = df[feature_columns].copy()

features["is_fraud"] = df["is_fraud"]

# lookup only, not model inputs
features["transaction_id"] = df["transaction_id"]

# needed for temporal train/test split
features["timestamp"] = df["timestamp"]


# basic validation
print(f"Total rows: {len(features):,}")
print(f"Features: {len(feature_columns)}")
print(f"Fraud rows: {features['is_fraud'].sum():,}")
print(f"Normal rows: {(features['is_fraud'] == 0).sum():,}")

print("\nMissing values:")
missing = features.isna().sum()
print(missing[missing > 0])


# save
OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)

features.to_csv(
    OUTPUT_PATH,
    index=False
)

print("\nSaved to:")
print(OUTPUT_PATH)