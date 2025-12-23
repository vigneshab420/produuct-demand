import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

st.set_page_config(page_title="Product Demand Segmentation", layout="centered")

st.title("📊 Product Demand Segmentation using K-Means")

st.subheader("Enter Product & Marketing Details")

# ---------------- USER INPUT ----------------
quantity_required = st.number_input("Quantity Required", min_value=0, value=100)
marketing_efforts = st.number_input("Marketing Efforts", min_value=0, value=50)
a_cost = st.number_input("Marketing Cost A", min_value=0.0, value=10000.0)
b_cost = st.number_input("Marketing Cost B", min_value=0.0, value=8000.0)
c_cost = st.number_input("Marketing Cost C", min_value=0.0, value=6000.0)

# ---------------- SAMPLE TRAINING DATA ----------------
# (Used to fit K-Means model)
data = {
    'quantity_required': [50, 200, 150, 80, 300, 120],
    'marketing_efforts': [30, 90, 70, 40, 110, 60],
    'a_marketing_annual_cost': [5000, 20000, 15000, 7000, 30000, 12000],
    'b_marketing_annual_cost': [4000, 18000, 13000, 6000, 25000, 10000],
    'c_marketing_annual_cost': [3000, 15000, 11000, 5000, 20000, 9000]
}

df = pd.DataFrame(data)

X = df[['quantity_required',
        'marketing_efforts',
        'a_marketing_annual_cost',
        'b_marketing_annual_cost',
        'c_marketing_annual_cost']]

# ---------------- SCALING ----------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ---------------- K-MEANS ----------------
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X_scaled)

# ---------------- PREDICTION ----------------
user_data_
