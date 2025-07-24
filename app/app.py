import streamlit as st
import os
from streamlit_option_menu import option_menu
import pandas as pd
import plotly.express as px
from sklearn.model_selection import train_test_split
import numpy as np 

# ================================
# 🔧 Utility to load cleaned dataset
# ================================
def load_data():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_dir, "..", "data", "cleaned_dataset.xlsx")
    return pd.read_excel(data_path)

# ✅ Load dataset
dataset = load_data()

# ================================
# ⚙️ Streamlit Page Config
# ================================
st.set_page_config(page_title="Customer Analysis", layout="wide")

# ================================
# 📌 Sidebar Navigation
# ================================
with st.sidebar:
    selected = option_menu(
        menu_title="Navigation",
        options=[
            "EDA Dashboard",
            "RFM Analysis",
            "Churn Modeling",
            "Spending Prediction"
        ],
        icons=[
            "bar-chart",       # EDA Dashboard
            "diagram-3",       # RFM Analysis
            "activity",        # Churn Modeling
            "cash-coin"        # Spending Prediction
        ],
        menu_icon="cast",
        default_index=0,
    )

# ================================
# 1️⃣ EDA DASHBOARD
# ================================
if selected == "EDA Dashboard":
    st.title("📊 Exploratory Data Analysis")

    st.subheader("Top 10 Countries by Revenue")
    dataset['Total_amount'] = dataset['Quantity'] * dataset['UnitPrice']
    country_rev = dataset.groupby('Country')['Total_amount'].sum().sort_values(ascending=False).head(10)
    st.plotly_chart(px.bar(country_rev, x=country_rev.index, y=country_rev.values, title="Revenue by Country"))

    st.subheader("Monthly Revenue Trend")
    dataset['InvoiceDate'] = pd.to_datetime(dataset['InvoiceDate'])
    dataset['Month'] = dataset['InvoiceDate'].dt.to_period('M')
    monthly_rev = dataset.groupby('Month')['Total_amount'].sum()
    st.line_chart(monthly_rev)

# ================================
# 2️⃣ RFM ANALYSIS
# ================================
elif selected == "RFM Analysis":
    st.title("📈 RFM Analysis")

    import datetime as dt
    snapshot_date = dataset['InvoiceDate'].max() + dt.timedelta(days=1)
    rfm = dataset.groupby('CustomerID').agg({
        'InvoiceDate': lambda x: (snapshot_date - x.max()).days,
        'InvoiceNo': 'nunique',
        'Total_amount': 'sum'
    })
    rfm.columns = ['Recency', 'Frequency', 'Monetary']
    st.dataframe(rfm.head())

    st.subheader("RFM Segmentation")
    rfm['R_Score'] = pd.qcut(rfm['Recency'], 4, labels=[4, 3, 2, 1])
    rfm['F_Score'] = pd.qcut(rfm['Frequency'].rank(method='first'), 4, labels=[1, 2, 3, 4])
    rfm['M_Score'] = pd.qcut(rfm['Monetary'], 4, labels=[1, 2, 3, 4])
    rfm['RFM_Score'] = rfm[['R_Score', 'F_Score', 'M_Score']].sum(axis=1).astype(int)
    st.dataframe(rfm[['Recency', 'Frequency', 'Monetary', 'RFM_Score']].head())

# ================================
# 3️⃣ CHURN MODELING
# ================================
elif selected == "Churn Modeling":
    st.title("🔍 Customer Churn Prediction")

    base_dir = os.path.dirname(os.path.abspath(__file__))
    rfm_path = os.path.join(base_dir, "..", "data", "RFM.xlsx")
    rfm_data = pd.read_excel(rfm_path)

    # ✅ Create churn label
    Churn_period = 180
    rfm_data['Churned'] = (rfm_data['Recency'] > Churn_period).astype('int64')

    X = rfm_data[['Recency', 'Frequency', 'Monetary']]
    y = rfm_data['Churned']

    from sklearn.linear_model import LogisticRegression
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=43)
    classification_model = LogisticRegression(class_weight='balanced', C=100, solver='liblinear')
    classification_model.fit(x_train, y_train)

    # Accuracy
    acc = classification_model.score(x_test, y_test)
    st.write(f"📈 **Churn Model Accuracy:** {acc:.4f}")

    st.subheader("Enter RFM Values:")
    rec_slider = st.slider("Recency (days since last purchase)", 0, 365, 30)
    rec = st.number_input("Enter Recency value:", 0, 365, rec_slider, step=1)
    freq_slider = st.slider("Frequency (number of purchases)", 0, 100, 5)
    freq = st.number_input("Enter Frequency value:", 0, 100, freq_slider, step=1)
    mon_slider = st.slider("Monetary (amount spent)", 0, 10000, 500)
    mon = st.number_input("Enter Monetary value:", 0, 10000, mon_slider, step=1)

    if st.button("Predict"):
        input_data = pd.DataFrame([[rec, freq, mon]], columns=['Recency', 'Frequency', 'Monetary'])
        prediction = classification_model.predict(input_data)[0]
        result = "Churned" if prediction == 1 else "Not Churned"
        st.success(f"✅ Prediction: **{result}**")

# ================================
# 4️⃣ ENHANCED SPENDING PREDICTION
# ================================
elif selected == "Spending Prediction":
    st.title("💰 Predict User Spending (Monetary)")

    # 🔹 Load your original dataset (not just pre-aggregated RFM)
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_dir, "..", "data", "cleaned_dataset.xlsx")
    try:
        raw_data = pd.read_excel(data_path)
    except FileNotFoundError:
        st.error(f"Error: cleaned_dataset.xlsx not found at {data_path}. Please ensure the file is in the correct directory.")
        st.stop()
    except Exception as e:
        st.error(f"Error loading cleaned_dataset.xlsx: {e}")
        st.stop()

    # ✅ Ensure InvoiceDate is datetime
    raw_data['InvoiceDate'] = pd.to_datetime(raw_data['InvoiceDate'])

    # ✅ Aggregate to RFM-like data per customer
    grouped = raw_data.groupby('CustomerID').agg(
        Recency=('InvoiceDate', lambda x: (raw_data['InvoiceDate'].max() - x.max()).days),
        Frequency=('InvoiceNo', 'nunique'),
        Monetary=('Total_amount', 'sum'),
        FirstPurchase=('InvoiceDate', 'min'),
        LastPurchase=('InvoiceDate', 'max')
    ).reset_index()

    # ✅ Compute tenure as (LastPurchase - FirstPurchase) in days
    grouped['Tenure'] = (grouped['LastPurchase'] - grouped['FirstPurchase']).dt.days
    grouped['Tenure'] = grouped['Tenure'].apply(lambda x: max(x, 1))  # Avoid zero/negative tenure

    # ✅ Features and Target
    feature_cols = ['Recency', 'Frequency', 'Tenure']
    X = grouped[feature_cols]
    y = grouped['Monetary']

    # ✅ Train-test split and model
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import r2_score, mean_absolute_error
    import numpy as np

    if len(X) < 2:
        st.error("Not enough data points to train the model.")
        st.stop()

    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    spend_model = LinearRegression()
    spend_model.fit(x_train, y_train)

    # ✅ Inputs for prediction
    st.subheader("Enter Customer Features to Predict Spending:")

    rec_max = int(X['Recency'].max()) if not X.empty else 365
    freq_max = int(X['Frequency'].max()) if not X.empty else 100
    tenure_max = int(X['Tenure'].max()) if not X.empty else 2000

    rec_input = st.slider("Recency (days since last purchase)", 0, rec_max, min(30, rec_max), key="spend_rec")
    freq_input = st.slider("Frequency (number of purchases)", 1, freq_max, min(5, freq_max), key="spend_freq")
    tenure_input = st.slider("Customer Tenure (days)", 1, tenure_max, min(365, tenure_max), key="spend_tenure")

    # ✅ Predict
    if st.button("Predict Spending"):
        input_data = pd.DataFrame([[rec_input, freq_input, tenure_input]], columns=feature_cols)
        predicted_spending = spend_model.predict(input_data)[0]
        predicted_spending = max(0, predicted_spending)
        st.success(f"💸 **Estimated Spending (Monetary): ${predicted_spending:.2f}**")
