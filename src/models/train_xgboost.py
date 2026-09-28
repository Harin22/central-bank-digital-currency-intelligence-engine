import pandas as pd
import joblib
import mlflow
import mlflow.sklearn

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report
)

from xgboost import XGBClassifier


df = pd.read_csv("data/processed/ml_features.csv")

df["timestamp"] = pd.to_datetime(df["timestamp"])
df = df.sort_values("timestamp").reset_index(drop=True)

print("Total transactions:", len(df))


#features
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


split = int(len(df) * 0.8)

X_train = X.iloc[:split]
X_test = X.iloc[split:]

y_train = y.iloc[:split]
y_test = y.iloc[split:]

print("Training rows:", len(X_train))
print("Testing rows :", len(X_test))

print("Training fraud rate:", round(y_train.mean(), 3))
print("Testing fraud rate :", round(y_test.mean(), 3))


#fill missing numeric values
numeric_transformer = SimpleImputer(
    strategy="median"
)

#categorical values to numbers
categorical_transformer = OneHotEncoder(
    handle_unknown="ignore"
)

preprocessor = ColumnTransformer(
    transformers=[
        ("numeric", numeric_transformer, numeric_features),
        ("categorical", categorical_transformer, categorical_features)
    ]
)


negative = y_train.value_counts()[0]
positive = y_train.value_counts()[1]

scale_pos_weight = negative / positive

print("Scale pos weight:", round(scale_pos_weight, 2))


#XGBoost core

n_estimators = 300
max_depth = 6
learning_rate = 0.05

model = XGBClassifier(
    n_estimators=n_estimators,
    max_depth=max_depth,
    learning_rate=learning_rate,
    scale_pos_weight=scale_pos_weight,
    objective="binary:logistic",
    eval_metric="logloss",
    random_state=42,
    n_jobs=-1
)


pipeline = Pipeline(
    steps=[
        ("preprocessing", preprocessor),
        ("model", model)
    ]
)


mlflow.set_experiment("CBDC Fraud Detection")

with mlflow.start_run(run_name="XGBoost"):

    print("\nTraining XGBoost...")

    pipeline.fit(X_train, y_train)

    # predictions
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    # metrics
    precision = precision_score(
        y_test,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_pred,
        zero_division=0
    )

    roc_auc = roc_auc_score(
        y_test,
        y_prob
    )

    pr_auc = average_precision_score(
        y_test,
        y_prob
    )

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
    print(
        classification_report(
            y_test,
            y_pred,
            zero_division=0
        )
    )

    # feature importance
    importance = pd.Series(
        pipeline.named_steps["model"].feature_importances_,
        index=pipeline.named_steps[
            "preprocessing"
        ].get_feature_names_out()
    ).sort_values(ascending=False)

    print("\nTop 15 features")
    print("--------------------")
    print(importance.head(15))

    mlflow.log_param("model", "XGBoost")
    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_param("max_depth", max_depth)
    mlflow.log_param("learning_rate", learning_rate)
    mlflow.log_param("scale_pos_weight", scale_pos_weight)
    mlflow.log_param("train_rows", len(X_train))
    mlflow.log_param("test_rows", len(X_test))

    # log results
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1_score", f1)
    mlflow.log_metric("roc_auc", roc_auc)
    mlflow.log_metric("pr_auc", pr_auc)

    # save model to MLflow
    mlflow.sklearn.log_model(
        pipeline,
        name="xgboost_model",
        serialization_format="cloudpickle"
    )


#save predictions
results = df.iloc[split:][
    ["transaction_id", "timestamp", "is_fraud"]
].copy()

results["prediction"] = y_pred
results["fraud_probability"] = y_prob

results.to_csv(
    "data/processed/xgb_predictions.csv",
    index=False
)


#save model locally
joblib.dump(
    pipeline,
    "models/xgboost.joblib"
)

print("\nSaved:")
print("Model       : models/xgboost.joblib")
print("Predictions : data/processed/xgb_predictions.csv")
print("MLflow      : XGBoost run logged")