import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Churn Analysis", page_icon="⚠️", layout="wide")

st.title("⚠️ Churn Prediction Dashboard")
st.markdown("XGBoost model — identify at-risk customers before they leave")

@st.cache_data
def load_churn():
    return pd.read_csv("data/churn_data.csv")

churn_df = load_churn()

st.divider()

# KPIs
col1, col2, col3, col4 = st.columns(4)
high_risk   = (churn_df["Risk_Segment"] == "High Risk").sum()
medium_risk = (churn_df["Risk_Segment"] == "Medium Risk").sum()
low_risk    = (churn_df["Risk_Segment"] == "Low Risk").sum()

with col1:
    st.metric("Total Customers", f"{len(churn_df):,}")
with col2:
    st.metric("High Risk", f"{high_risk:,}", delta="Immediate action needed", delta_color="inverse")
with col3:
    st.metric("Medium Risk", f"{medium_risk:,}", delta="Monitor closely", delta_color="off")
with col4:
    st.metric("Low Risk", f"{low_risk:,}", delta="Stable", delta_color="normal")

st.divider()

# Risk distribution
st.subheader("📊 Churn Risk Distribution")
col1, col2 = st.columns(2)

with col1:
    risk_counts = churn_df["Risk_Segment"].value_counts()
    fig_bar = px.bar(
        x=risk_counts.index,
        y=risk_counts.values,
        color=risk_counts.index,
        color_discrete_map={
            "High Risk"  : "#e74c3c",
            "Medium Risk": "#f39c12",
            "Low Risk"   : "#2ecc71"
        },
        labels={"x": "Risk Segment", "y": "Customer Count"}
    )
    fig_bar.update_layout(
        height=350,
        showlegend=False,
        margin=dict(l=0, r=0, t=30, b=0)
    )
    st.plotly_chart(fig_bar, use_container_width=True)

with col2:
    fig_pie = px.pie(
        values=risk_counts.values,
        names=risk_counts.index,
        color=risk_counts.index,
        color_discrete_map={
            "High Risk"  : "#e74c3c",
            "Medium Risk": "#f39c12",
            "Low Risk"   : "#2ecc71"
        },
        hole=0.4
    )
    fig_pie.update_layout(
        height=350,
        margin=dict(l=0, r=0, t=30, b=0)
    )
    st.plotly_chart(fig_pie, use_container_width=True)

st.divider()

# Customer list by risk
st.subheader("🔎 Customer List by Risk Level")
risk_filter = st.selectbox(
    "Filter by Risk",
    ["All", "High Risk", "Medium Risk", "Low Risk"]
)

if risk_filter == "All":
    display_df = churn_df
else:
    display_df = churn_df[churn_df["Risk_Segment"] == risk_filter]

cols_show = ["Customer ID", "Frequency", "Monetary",
             "Tenure", "Churn_Probability", "Risk_Segment"]

st.dataframe(
    display_df[cols_show]
    .sort_values("Churn_Probability", ascending=False)
    .head(100)
    .style.format({
        "Monetary"         : "£{:,.2f}",
        "Churn_Probability": "{:.2%}"
    }),
    use_container_width=True,
    hide_index=True
)

st.divider()

# Model info
st.subheader("🤖 Model Information")
col1, col2 = st.columns(2)
with col1:
    st.markdown("""
    **Model:** XGBoost Classifier  
    **AUC-ROC:** 0.80
    **Accuracy:** 73%  
    **Features:** 11 behavioral features  
    **Churn Definition:** No purchase in 90+ days  
    """)
with col2:
    st.info("""
    **Note on AUC:** Target was 0.88 but achieved 0.80 
    without data leakage. Recency feature was excluded 
    as it directly encodes the churn label — using it 
    would give fake 100% AUC.
    """)