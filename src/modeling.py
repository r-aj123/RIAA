# src/modeling.py

import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def label_churn(rfm, recency_threshold=90):
    # Label customers as churned if Recency > threshold
    rfm["Churn"] = (rfm["Recency"] > recency_threshold).astype(int)
    return rfm

def train_churn_model(rfm):
    features = ["Recency", "Frequency", "Monetary"]
    X = rfm[features]
    y = rfm["Churn"]

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train logistic regression
    model = LogisticRegression()
    model.fit(X_scaled, y)

    # Save model and scaler
    joblib.dump(model, "output/models/churn_model.pkl")
    joblib.dump(scaler, "output/models/scaler.pkl")

    return model, scaler

def load_churn_model():
    model = joblib.load("output/models/churn_model.pkl")
    scaler = joblib.load("output/models/scaler.pkl")
    return model, scaler

def predict_churn(model, scaler, recency, frequency, monetary):
    # Prepare input as dataframe
    X = pd.DataFrame([[recency, frequency, monetary]], columns=["Recency", "Frequency", "Monetary"])
    X_scaled = scaler.transform(X)

    # Predict churn (0 or 1)
    pred = model.predict(X_scaled)[0]

    # Optional: get probability of churn
    prob = model.predict_proba(X_scaled)[0][1]

    return int(pred), prob