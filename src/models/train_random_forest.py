import os

import pandas as pd
import joblib
import mlflow
import mlflow.sklearn

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report
)


# make sure output folders exist
os.makedirs("models", exist_ok=True)
os.makedirs("data/processed", exist_ok=True)


# load data
df = pd.read_csv("data/processed/ml_features.csv")

df["timestamp"] = pd.to_datetime(df["timestamp"])
df = df.sort_values("timestamp").reset_index(drop=True)

print("Total transactions:", len(df))


# features
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
    "is_weekend"
]

categorical_features = [
    "transaction_type",
    "payment_mode",
    "country",
    "merchant_category"
]

features = numeric_features + categorical_features

X = df[features]
y = df["is_fraud"]


# temporal train/test split
split = int(len(df) * 0.8)

X_train = X.iloc[:split]
X_test = X.iloc[split:]

y_train = y.iloc[:split]
y_test = y.iloc[split:]

print("Training rows:", len(X_train))
print("Testing rows :", len(X_test))

print("Training fraud rate:", round(y_train.mean(), 3))
print("Testing fraud rate :", round(y_test.mean(), 3))


# handle missing numeric values
numeric_transformer = SimpleImputer(
    strategy="median",
    add_indicator=True
)

# convert categorical values to numbers
categorical_transformer = OneHotEncoder(
    handle_unknown="ignore"
)

preprocessor = ColumnTransformer(
    transformers=[
        ("numeric", numeric_transformer, numeric_features),
        ("categorical", categorical_transformer, categorical_features)
    ]
)


# random forest
n_estimators = 300
min_samples_leaf = 2

model = RandomForestClassifier(
    n_estimators=n_estimators,
    min_samples_leaf=min_samples_leaf,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

pipeline = Pipeline(
    steps=[
        ("preprocessing", preprocessor),
        ("model", model)
    ]
)


# start MLflow experiment
mlflow.set_experiment("CBDC Fraud Detection")

with mlflow.start_run(run_name="Random Forest"):

    print("\nTraining Random Forest...")

    pipeline.fit(X_train, y_train)

    # predictions
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    # metrics
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_prob)
    pr_auc = average_precision_score(y_test, y_prob)

    print("\nModel results")
    print("--------------------")
    print("Precision :", round(precision, 4))
    print("Recall    :", round(recall, 4))
    print("F1 score  :", round(f1, 4))
    print("ROC-AUC   :", round(roc_auc, 4))
    print("PR-AUC    :", round(pr_auc, 4))

    print("\nConfusion matrix")
    print(confusion_matrix(y_test, y_pred))

    print("\nClassification report")
    print(classification_report(y_test, y_pred, zero_division=0))

    # feature importance
    importance = pd.Series(
        pipeline.named_steps["model"].feature_importances_,
        index=pipeline.named_steps["preprocessing"].get_feature_names_out()
    ).sort_values(ascending=False)

    print("\nTop 15 features")
    print("--------------------")
    print(importance.head(15))

    # log model settings
    mlflow.log_param("model", "RandomForest")
    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_param("min_samples_leaf", min_samples_leaf)
    mlflow.log_param("class_weight", "balanced")
    mlflow.log_param("train_rows", len(X_train))
    mlflow.log_param("test_rows", len(X_test))

    # log results
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1_score", f1)
    mlflow.log_metric("roc_auc", roc_auc)
    mlflow.log_metric("pr_auc", pr_auc)

    # save model to MLflow (cloudpickle avoids the skops "untrusted types" error)
    mlflow.sklearn.log_model(
        pipeline,
        name="random_forest_model",
        serialization_format="cloudpickle"
    )


# save predictions
results = df.iloc[split:][
    ["transaction_id", "timestamp", "is_fraud"]
].copy()

results["prediction"] = y_pred
results["fraud_probability"] = y_prob

results.to_csv(
    "data/processed/rf_predictions.csv",
    index=False
)

# save model locally too
joblib.dump(
    pipeline,
    "models/random_forest.joblib"
)

print("\nSaved:")
print("Model       : models/random_forest.joblib")
print("Predictions : data/processed/rf_predictions.csv")
print("MLflow      : Random Forest run logged")