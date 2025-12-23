import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

st.set_page_config(page_title="Product Demand Segmentation", layout="centered")

st.title("📊 Product Demand Segmentation using K-Means")

# Upload CSV file
uploaded_file = st.file_uploader("Upload your CSV dataset", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.subheader("📁 Dataset Preview")
    st.dataframe(df.head())

    # Convert cost columns (remove 'R' if present)
    cost_cols = [
        'a_marketing_annual_cost',
        'b_marketing_annual_cost',
        'c_marketing_annual_cost'
    ]

    for col in cost_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.replace('R', '', regex=False)
            df[col] = df[col].astype(float)

    # Feature selection
    X = df[['quantity_required',
            'marketing_efforts',
            'a_marketing_annual_cost',
            'b_marketing_annual_cost',
            'c_marketing_annual_cost']]

    # Scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # K-Means
    kmeans = KMeans(n_clusters=3, random_state=42)
    df['Demand_Segment'] = kmeans.fit_predict(X_scaled)

    st.subheader("✅ Segmented Output")
    st.dataframe(df[['quantity_required', 'marketing_efforts', 'Demand_Segment']].head())

    # Visualization
    st.subheader("📈 Demand Segmentation Visualization")
    fig, ax = plt.subplots()
    scatter = ax.scatter(
        df['quantity_required'],
        df['marketing_efforts'],
        c=df['Demand_Segment']
    )
    ax.set_xlabel("Quantity Required")
    ax.set_ylabel("Marketing Efforts")
    ax.set_title("Product Demand Segmentation")
    st.pyplot(fig)

    st.markdown("""
    ### 🔍 Cluster Meaning
    - **0 → High Demand**
    - **1 → Medium Demand**
    - **2 → Low Demand**
    """)
