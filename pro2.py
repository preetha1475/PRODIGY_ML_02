# =====================================================
# Customer Segmentation using K-Means (Mall Dataset)
# =====================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import silhouette_score

# -----------------------------------------------------
# Page Config + Styling
# -----------------------------------------------------
st.set_page_config(page_title="Customer Segmentation", layout="wide")

st.markdown("""
<style>
.main {background: linear-gradient(135deg, #f5f7fa, #e4ecf7);}
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #232526, #414345);
}
section[data-testid="stSidebar"] * {color: white;}
.card {
    background: white;
    padding: 18px;
    border-radius: 14px;
    box-shadow: 0 6px 16px rgba(0,0,0,0.12);
}
h1, h2, h3 {color:#1f3b4d;}
</style>
""", unsafe_allow_html=True)

st.title("🛍️ Customer Segmentation using K-Means")

# -----------------------------------------------------
# Load / Create Dataset (Your Provided Data)
# -----------------------------------------------------
@st.cache_data
def load_data():
    return pd.DataFrame({
        "CustomerID": [1,2,3,4,5,6,7,8,9,10],
        "Gender": ["Male","Male","Female","Female","Female","Female","Female","Female","Male","Female"],
        "Age": [19,21,20,23,31,22,35,23,64,30],
        "Annual Income (k$)": [15,15,16,16,17,17,18,18,19,19],
        "Spending Score (1-100)": [39,81,6,77,40,76,6,94,3,72]
    })

data = load_data()

# -----------------------------------------------------
# Encode Gender
# -----------------------------------------------------
encoder = LabelEncoder()
data["GenderEncoded"] = encoder.fit_transform(data["Gender"])

# -----------------------------------------------------
# Sidebar Controls
# -----------------------------------------------------
st.sidebar.header("⚙️ Clustering Settings")

k = st.sidebar.slider("Number of Clusters (K)", 2, 5, 3)

features = st.sidebar.multiselect(
    "Select Features for Clustering",
    ["Age", "Annual Income (k$)", "Spending Score (1-100)", "GenderEncoded"],
    default=["Annual Income (k$)", "Spending Score (1-100)"]
)

# -----------------------------------------------------
# Preprocessing
# -----------------------------------------------------
X = data[features]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# -----------------------------------------------------
# K-Means Model
# -----------------------------------------------------
kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
data["Cluster"] = kmeans.fit_predict(X_scaled)

silhouette = silhouette_score(X_scaled, data["Cluster"])

# -----------------------------------------------------
# KPI Cards
# -----------------------------------------------------
c1, c2, c3 = st.columns(3)

with c1:
    st.markdown(f"""
    <div class="card">
        <h3>👥 Customers</h3>
        <h2>{len(data)}</h2>
    </div>
    """, unsafe_allow_html=True)

with c2:
    st.markdown(f"""
    <div class="card">
        <h3>🔢 Clusters</h3>
        <h2>{k}</h2>
    </div>
    """, unsafe_allow_html=True)

with c3:
    st.markdown(f"""
    <div class="card">
        <h3>📊 Silhouette Score</h3>
        <h2>{silhouette:.3f}</h2>
    </div>
    """, unsafe_allow_html=True)

# -----------------------------------------------------
# Cluster Visualization
# -----------------------------------------------------
st.markdown("## 📈 Cluster Visualization")

if len(features) >= 2:
    fig = px.scatter(
        data,
        x=features[0],
        y=features[1],
        color="Cluster",
        hover_data=["CustomerID", "Gender"],
        title=f"{features[0]} vs {features[1]}",
        template="plotly_white"
    )
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Please select at least two features.")

# -----------------------------------------------------
# Cluster Profiles
# -----------------------------------------------------
st.markdown("## 🧠 Cluster Profiles")

profile = data.groupby("Cluster")[features].mean().round(2)
st.dataframe(profile)

fig = px.bar(
    profile.reset_index().melt(id_vars="Cluster"),
    x="value",
    y="variable",
    color="Cluster",
    orientation="h",
    title="Average Feature Values per Cluster",
    template="plotly_white"
)

st.plotly_chart(fig, use_container_width=True)

# -----------------------------------------------------
# Elbow Method
# -----------------------------------------------------
st.markdown("## 📉 Elbow Method")

wcss = []
K_range = range(2, 7)
for i in K_range:
    km = KMeans(n_clusters=i, random_state=42, n_init=10)
    km.fit(X_scaled)
    wcss.append(km.inertia_)

elbow_df = pd.DataFrame({"K": list(K_range), "WCSS": wcss})

fig = px.line(
    elbow_df,
    x="K",
    y="WCSS",
    markers=True,
    title="Elbow Curve",
    template="plotly_white"
)

st.plotly_chart(fig, use_container_width=True)

# -----------------------------------------------------
# View Dataset
# -----------------------------------------------------
with st.expander("📂 View Clustered Dataset"):
    st.dataframe(data)
