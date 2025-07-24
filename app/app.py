import streamlit as st
import os
from streamlit_option_menu import option_menu
import pandas as pd
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import datetime as dt

# Load dataset
def load_data():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_dir, "..", "data", "cleaned_dataset.xlsx")
    return pd.read_excel(data_path)

dataset = load_data()

# Set page config
st.set_page_config(page_title="Customer Analysis", layout="wide")

# 💠 Sidebar Layout
with st.sidebar:
    st.markdown("## Customer Analytics Dashboard")
    st.markdown("Segment, Analyze, Predict")
    st.markdown("---")

    selected = option_menu(
        menu_title="📂 Dashboard Sections",
        options=["EDA Dashboard", "RFM Analysis", "Churn Modeling"],
        icons=["bar-chart", "layers", "graph-up"],
        menu_icon="grid",
        default_index=0,
    )

    st.markdown("---")

    if selected == "EDA Dashboard":
        st.markdown("Gain insights from transaction patterns and trends.")
    elif selected == "RFM Analysis":
        st.markdown("Segment customers using Recency, Frequency, and Monetary value.")
    elif selected == "Churn Modeling":
        st.markdown("Predict likelihood of customer churn using RFM data.")

    st.markdown("---")
    st.markdown("#### Dataset Overview")
    st.metric("Total Records", len(dataset))
    st.metric("Unique Customers", dataset['CustomerID'].nunique())

    st.markdown("---")
    st.caption("Built by: Your Team Name")
    st.caption("CS Project · 2025")

# ✅ Common Preprocessing
dataset['Total_amount'] = dataset['Quantity'] * dataset['UnitPrice']
dataset['InvoiceDate'] = pd.to_datetime(dataset['InvoiceDate'])

# 1️⃣ EDA DASHBOARD
if selected == "EDA Dashboard":
    st.title("📊 Exploratory Data Analysis")

    # Key Metrics Section
    total_revenue = dataset['Total_amount'].sum()
    total_customers = dataset['CustomerID'].nunique()
    total_orders = dataset['InvoiceNo'].nunique()

    st.markdown("### 🔢 Key Metrics")
    kpi1, kpi2, kpi3 = st.columns(3)
    kpi1.metric("Total Revenue", f"${total_revenue:,.0f}")
    kpi2.metric("Unique Customers", total_customers)
    kpi3.metric("Total Orders", total_orders)

    # Revenue by Hour
    st.subheader("Revenue by Hour of Day")
    dataset['Hour'] = dataset['InvoiceDate'].dt.hour
    hourly_rev = dataset.groupby('Hour')['Total_amount'].sum().reset_index()
    fig_hourly = px.bar(hourly_rev, x='Hour', y='Total_amount',
                        title='Revenue by Hour of Day',
                        labels={'Hour': 'Hour (0–23)', 'Total_amount': 'Revenue'})
    st.plotly_chart(fig_hourly, use_container_width=True)

    # Monthly Revenue Trend
    st.subheader("Monthly Revenue Trend")
    dataset['Month'] = dataset['InvoiceDate'].dt.to_period('M').astype(str)
    monthly_rev = dataset.groupby('Month')['Total_amount'].sum().reset_index()
    fig_monthly = px.line(monthly_rev, x='Month', y='Total_amount',
                          title='Monthly Revenue Over Time',
                          labels={'Total_amount': 'Revenue'})
    fig_monthly.update_traces(mode='lines+markers')
    fig_monthly.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig_monthly)

# 2️⃣ RFM ANALYSIS
elif selected == "RFM Analysis":
    st.title("📈 RFM (Recency, Frequency, Monetary) Analysis")

    snapshot_date = dataset['InvoiceDate'].max() + dt.timedelta(days=1)
    rfm = dataset.groupby('CustomerID').agg({
        'InvoiceDate': lambda x: (snapshot_date - x.max()).days,
        'InvoiceNo': 'nunique',
        'Total_amount': 'sum'
    }).reset_index()
    rfm.columns = ['CustomerID', 'Recency', 'Frequency', 'Monetary']

    # 👉 RFM Scores
    rfm['R_Score'] = pd.qcut(rfm['Recency'], 4, labels=[4, 3, 2, 1])
    rfm['F_Score'] = pd.qcut(rfm['Frequency'].rank(method='first'), 4, labels=[1, 2, 3, 4])
    rfm['M_Score'] = pd.qcut(rfm['Monetary'], 4, labels=[1, 2, 3, 4])
    rfm['RFM_Score'] = rfm[['R_Score', 'F_Score', 'M_Score']].sum(axis=1).astype(int)

    # 🔢 Summary Metrics
    st.markdown("### 🧮 RFM Summary Metrics")
    col1, col2, col3 = st.columns(3)
    col1.metric("Avg. Recency (days)", f"{rfm['Recency'].mean():.0f}")
    col2.metric("Avg. Frequency", f"{rfm['Frequency'].mean():.1f}")
    col3.metric("Avg. Monetary", f"${rfm['Monetary'].mean():,.0f}")

    # 📊 Distribution of RFM Score
    st.markdown("### 📊 RFM Score Distribution")
    fig_rfm_score = px.histogram(
        rfm, x='RFM_Score', nbins=10,
        title='Distribution of RFM Scores',
        color_discrete_sequence=["#636EFA"]
    )
    st.plotly_chart(fig_rfm_score, use_container_width=True)

    # 🔍 Detailed Table in Expander
    with st.expander("🔽 View Raw RFM Table"):
        st.dataframe(rfm.head(10))

    with st.expander("📈 Top Customers by RFM Score"):
        top_customers = rfm.sort_values("RFM_Score", ascending=False).head(10)
        st.dataframe(top_customers.style.background_gradient(cmap='Blues'))

    st.markdown("---")
    st.caption("RFM segmentation helps understand which customers are the most valuable based on how recently, how often, and how much they purchase.")

# 3️⃣ CHURN MODELING
elif selected == "Churn Modeling":
    st.title("🔍 Customer Churn Prediction")

    base_dir = os.path.dirname(os.path.abspath(__file__))
    rfm_path = os.path.join(base_dir, "..", "data", "RFM.xlsx")
    rfm_data = pd.read_excel(rfm_path)

    # Define churn condition
    Churn_period = 180
    rfm_data['Churned'] = (rfm_data['Recency'] > Churn_period).astype(int)

    X = rfm_data[['Recency', 'Frequency', 'Monetary']]
    y = rfm_data['Churned']
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=43)

    classification_model = LogisticRegression(class_weight='balanced', C=100, solver='liblinear')
    classification_model.fit(x_train, y_train)

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
