import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="Segmentation", page_icon="👥", layout="wide")

st.title("👥 Customer Segmentation")
st.markdown("RFM-based KMeans clustering — Champions, Loyal, At-Risk customers")

@st.cache_data
def load_rfm():
    return pd.read_csv("data/rfm_data.csv")

rfm = load_rfm()

st.divider()

# Segment distribution
st.subheader("📊 Customer Segment Distribution")
col1, col2 = st.columns([1, 2])

with col1:
    seg_counts = rfm["Segment"].value_counts()
    fig_pie = px.pie(
        values=seg_counts.values,
        names=seg_counts.index,
        color_discrete_sequence=["#2ecc71", "#3498db", "#e74c3c"],
        hole=0.4
    )
    fig_pie.update_layout(height=350, margin=dict(l=0, r=0, t=30, b=0))
    st.plotly_chart(fig_pie, use_container_width=True)

with col2:
    seg_stats = rfm.groupby("Segment").agg(
        Customers = ("Customer ID", "count"),
        Avg_Recency  = ("Recency", "mean"),
        Avg_Frequency= ("Frequency", "mean"),
        Avg_Monetary = ("Monetary", "mean")
    ).round(2).reset_index()
    st.dataframe(seg_stats, use_container_width=True, hide_index=True)

st.divider()

# RFM Scatter
st.subheader("🔍 RFM Scatter Plot")
col1, col2 = st.columns(2)

with col1:
    x_axis = st.selectbox("X Axis", ["Recency", "Frequency", "Monetary"], index=0)
with col2:
    y_axis = st.selectbox("Y Axis", ["Monetary", "Frequency", "Recency"], index=0)

fig_scatter = px.scatter(
    rfm,
    x=x_axis, y=y_axis,
    color="Segment",
    hover_data=["Customer ID", "Recency", "Frequency", "Monetary"],
    color_discrete_map={
        "Champions"    : "#2ecc71",
        "Loyal Customers": "#3498db",
        "At-Risk / Lost": "#e74c3c"
    },
    opacity=0.6
)
fig_scatter.update_layout(
    height=450,
    margin=dict(l=0, r=0, t=30, b=0)
)
st.plotly_chart(fig_scatter, use_container_width=True)

st.divider()

# Segment filter
st.subheader("🔎 Customer List by Segment")
selected_seg = st.selectbox(
    "Select Segment",
    ["All"] + rfm["Segment"].unique().tolist()
)

if selected_seg == "All":
    filtered = rfm
else:
    filtered = rfm[rfm["Segment"] == selected_seg]

st.dataframe(
    filtered[["Customer ID", "Recency", "Frequency", "Monetary", "Segment"]]
    .sort_values("Monetary", ascending=False)
    .head(100),
    use_container_width=True,
    hide_index=True
)