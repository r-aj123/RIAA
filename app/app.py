import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import plotly.express as px

# Load dataset (modify the path if required)
@st.cache_data
def load_data():
    df = pd.read_excel(r'C:\Users\kraja\OneDrive\Desktop\RIAA-1\data\cleaned_dataset.xlsx')  # adjust this as needed
    return df

dataset = load_data()

# Navbar
st.set_page_config(page_title="Customer Analysis", layout="wide")

with st.sidebar:
    selected = option_menu(
        menu_title="Navigation",
        options=["EDA Dashboard", "RFM Analysis", "Churn Modeling"],
        icons=["bar-chart", "diagram-3", "activity"],
        menu_icon="cast",
        default_index=0,
    )

# 1️⃣ EDA DASHBOARD
if selected == "EDA Dashboard":
    st.title("📊 Exploratory Data Analysis")

    st.subheader("Top 10 Countries by Revenue")
    dataset['TotalAmount'] = dataset['Quantity'] * dataset['UnitPrice']
    country_rev = dataset.groupby('Country')['TotalAmount'].sum().sort_values(ascending=False).head(10)
    st.plotly_chart(px.bar(country_rev, x=country_rev.index, y=country_rev.values, title="Revenue by Country"))

    st.subheader("Monthly Revenue Trend")
    dataset['InvoiceDate'] = pd.to_datetime(dataset['InvoiceDate'])
    dataset['Month'] = dataset['InvoiceDate'].dt.to_period('M')
    monthly_rev = dataset.groupby('Month')['TotalAmount'].sum()
    st.line_chart(monthly_rev)

# 2️⃣ RFM ANALYSIS
elif selected == "RFM Analysis":
    st.title("📈 RFM Analysis")

    import datetime as dt

    snapshot_date = dataset['InvoiceDate'].max() + dt.timedelta(days=1)
    rfm = dataset.groupby('CustomerID').agg({
        'InvoiceDate': lambda x: (snapshot_date - x.max()).days,
        'InvoiceNo': 'nunique',
        'TotalAmount': 'sum'
    })
    rfm.columns = ['Recency', 'Frequency', 'Monetary']
    st.dataframe(rfm.head())

    st.subheader("RFM Segmentation")
    rfm['R_Score'] = pd.qcut(rfm['Recency'], 4, labels=[4, 3, 2, 1])
    rfm['F_Score'] = pd.qcut(rfm['Frequency'].rank(method='first'), 4, labels=[1, 2, 3, 4])
    rfm['M_Score'] = pd.qcut(rfm['Monetary'], 4, labels=[1, 2, 3, 4])
    rfm['RFM_Score'] = rfm[['R_Score', 'F_Score', 'M_Score']].sum(axis=1).astype(int)
    st.dataframe(rfm[['Recency', 'Frequency', 'Monetary', 'RFM_Score']].head())

# 3️⃣ CHURN MODELING
elif selected == "Churn Modeling":
    st.title("🔍 Customer Churn Prediction")

    st.subheader("Enter RFM Values:")
    rec = st.slider("Recency", min_value=0, max_value=365, value=30)
    freq = st.slider("Frequency", min_value=0, max_value=100, value=5)
    mon = st.slider("Monetary", min_value=0, max_value=10000, value=500)

    # Dummy Logistic Regression (replace with real model later)
    import numpy as np
    from sklearn.linear_model import LogisticRegression

    # Train a simple model for demonstration
    rfm['Churned'] = np.where(rfm['RFM_Score'] <= 6, 1, 0)
    X = rfm[['Recency', 'Frequency', 'Monetary']]
    y = rfm['Churned']
    model = LogisticRegression()
    model.fit(X, y)

    # Predict churn
    input_data = pd.DataFrame([[rec, freq, mon]], columns=['Recency', 'Frequency', 'Monetary'])
    prediction = model.predict(input_data)[0]
    result = "Churned" if prediction == 1 else "Not Churned"
    st.success(f"Prediction: **{result}**")
