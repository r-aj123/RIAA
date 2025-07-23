import pandas as pd
from datetime import timedelta

def load_and_clean_data(filepath):
    df = pd.read_excel(filepath)

    # Drop missing customers
    df.dropna(subset=["CustomerID"], inplace=True)

    # Remove returns and cancelled orders
    df = df[df["Quantity"] > 0]
    df = df[~df["InvoiceNo"].astype(str).str.startswith("C")]

    # Convert date and add total price
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    df["TotalPrice"] = df["Quantity"] * df["UnitPrice"]

    return df

def calculate_rfm(df):
    snapshot_date = df["InvoiceDate"].max() + timedelta(days=1)

    rfm = df.groupby("CustomerID").agg({
        "InvoiceDate": lambda x: (snapshot_date - x.max()).days,
        "InvoiceNo": "nunique",
        "TotalPrice": "sum"
    }).reset_index()

    rfm.columns = ["CustomerID", "Recency", "Frequency", "Monetary"]

    return rfm