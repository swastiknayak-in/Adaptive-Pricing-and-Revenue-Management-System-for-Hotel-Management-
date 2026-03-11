# model_training.py
# Train pricing model for Adaptive Pricing and Revenue Management System

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import joblib


# -----------------------------
# Load dataset
# -----------------------------
def load_data(path="data/hotel_bookings.csv"):
    return pd.read_csv(path)


# -----------------------------
# Convert month name → number
# -----------------------------
def month_to_num(col):

    months = {
        "January":1,"February":2,"March":3,"April":4,
        "May":5,"June":6,"July":7,"August":8,
        "September":9,"October":10,"November":11,"December":12
    }

    return col.map(months)


# -----------------------------
# Feature preprocessing
# -----------------------------
def preprocess_features(df):

    df = df.copy()

    df["arrival_month"] = month_to_num(df["arrival_date_month"])

    df["demand_score"] = (
        df["lead_time"] +
        df["previous_bookings_not_canceled"]
    )

    df = df.dropna(subset=["adr"])

    X = df[
        [
            "hotel",
            "lead_time",
            "arrival_month",
            "reserved_room_type",
            "customer_type",
            "previous_bookings_not_canceled",
            "demand_score"
        ]
    ]

    y = df["adr"]

    return X, y


# -----------------------------
# Train model
# -----------------------------
def train_and_save(data_path="data/hotel_bookings.csv", model_dir="models"):

    os.makedirs(model_dir, exist_ok=True)

    df = load_data(data_path)

    X, y = preprocess_features(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    numeric_features = [
        "lead_time",
        "arrival_month",
        "previous_bookings_not_canceled",
        "demand_score"
    ]

    categorical_features = [
        "hotel",
        "reserved_room_type",
        "customer_type"
    ]

    numeric_transformer = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]
    )

    categorical_transformer = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore"))
        ]
    )

    preprocessor = ColumnTransformer(
        [
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features)
        ]
    )

    model = RandomForestRegressor(
        n_estimators=150,
        random_state=42
    )

    pipeline = Pipeline(
        [
            ("preprocessor", preprocessor),
            ("model", model)
        ]
    )

    pipeline.fit(X_train, y_train)

    preds = pipeline.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, preds))

    print("Model RMSE:", rmse)

    model_path = os.path.join(model_dir, "pricing_model.pkl")

    joblib.dump(pipeline, model_path)

    print("Model saved:", model_path)

    return pipeline


if __name__ == "__main__":
    train_and_save()
