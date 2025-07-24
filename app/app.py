import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import plotly.express as px
from sklearn.model_selection import train_test_split

# Load dataset (modify the path if required)
@st.cache_data
def load_data():
    df = pd.read_excel(r'C:\Users\atuls\OneDrive\Desktop\RIAA-2\data\cleaned_dataset.xlsx')  # adjust this as needed
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
    dataset['Total_amount'] = dataset['Quantity'] * dataset['UnitPrice']
    country_rev = dataset.groupby('Country')['Total_amount'].sum().sort_values(ascending=False).head(10)
    st.plotly_chart(px.bar(country_rev, x=country_rev.index, y=country_rev.values, title="Revenue by Country"))

    st.subheader("Monthly Revenue Trend")
    dataset['InvoiceDate'] = pd.to_datetime(dataset['InvoiceDate'])
    dataset['Month'] = dataset['InvoiceDate'].dt.to_period('M')
    monthly_rev = dataset.groupby('Month')['Total_amount'].sum()
    st.line_chart(monthly_rev)

# 2️⃣ RFM ANALYSIS
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

# 3️⃣ CHURN MODELING
elif selected == "Churn Modeling":
    st.title("🔍 Customer Churn Prediction")

    # 👉 Load the same RFM dataset you used in the notebook
    rfm_data = pd.read_excel(r"C:\Users\atuls\OneDrive\Desktop\RIAA-2\data\RFM.xlsx")

    # 👉 Apply the same churn definition
    Churn_period = 180
    rfm_data['Churned'] = (rfm_data['Recency'] > Churn_period).astype('int64')

    # 👉 Prepare features and target
    X = rfm_data[['Recency', 'Frequency', 'Monetary']]
    y = rfm_data['Churned']

    # 👉 Split the data like in the notebook
    from sklearn.model_selection import train_test_split
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=43)

    # 👉 Train the same logistic regression model
    from sklearn.linear_model import LogisticRegression
    classification_model = LogisticRegression(
        class_weight='balanced',
        C=100,
        solver='liblinear'
    )
    classification_model.fit(x_train, y_train)

st.subheader("Enter RFM Values:")

# 🔹 Recency
rec_slider = st.slider("Recency (days since last purchase)", min_value=0, max_value=365, value=30)
rec = st.number_input("Enter Recency value:", min_value=0, max_value=365, value=rec_slider, step=1)

# 🔹 Frequency
freq_slider = st.slider("Frequency (number of purchases)", min_value=0, max_value=100, value=5)
freq = st.number_input("Enter Frequency value:", min_value=0, max_value=100, value=freq_slider, step=1)


# 🔹 Monetary
mon_slider = st.slider("Monetary (amount spent)", min_value=0, max_value=10000, value=500)
mon = st.number_input("Enter Monetary value:", min_value=0, max_value=10000, value=mon_slider, step=1)


    # 👉 Predict button
if st.button("Predict"):
    input_data = pd.DataFrame(
        [[rec, freq, mon]],
            columns=['Recency', 'Frequency', 'Monetary']
    )
    prediction = classification_model.predict(input_data)[0]
    result = "Churned" if prediction == 1 else "Not Churned"
    st.success(f"✅ Prediction: **{result}**")
