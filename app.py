import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(
    page_title="RetailPulse",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f77b4;
        text-align: center;
        padding: 1rem 0;
    }
    .metric-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
    }
    .subtitle {
        font-size: 1.1rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<p class="main-header">📊 RetailPulse</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">AI-Powered Customer Analytics & Demand Forecasting Platform</p>', 
            unsafe_allow_html=True)

st.divider()

# Data load karo
@st.cache_data
def load_data():
    daily_sales = pd.read_csv("data/daily_sales.csv", parse_dates=["ds"])
    churn_data  = pd.read_csv("data/churn_data.csv")
    rfm_data    = pd.read_csv("data/rfm_data.csv")
    inventory   = pd.read_csv("data/inventory_plan.csv", parse_dates=["Date"])
    forecast    = pd.read_csv("data/forecast_comparison.csv", parse_dates=["Date"])
    return daily_sales, churn_data, rfm_data, inventory, forecast

daily_sales, churn_data, rfm_data, inventory, forecast = load_data()

# KPIs — top row
st.subheader("📈 Business Overview")
col1, col2, col3, col4 = st.columns(4)

with col1:
    total_revenue = daily_sales["y"].sum()
    st.metric(
        label="Total Revenue",
        value=f"£{total_revenue:,.0f}",
        delta="2 Years Data"
    )

with col2:
    total_customers = len(rfm_data)
    st.metric(
        label="Total Customers",
        value=f"{total_customers:,}",
        delta="UK Market"
    )

with col3:
    churn_rate = churn_data["Churned"].mean() * 100
    st.metric(
        label="Churn Rate",
        value=f"{churn_rate:.1f}%",
        delta="180 day threshold",
        delta_color="inverse"
    )

with col4:
    avg_daily = daily_sales["y"].mean()
    st.metric(
        label="Avg Daily Revenue",
        value=f"£{avg_daily:,.0f}",
        delta="Per trading day"
    )

st.divider()

# Revenue trend — simple overview graph
st.subheader("📉 Revenue Trend (2009–2011)")
import plotly.express as px

fig = px.line(
    daily_sales,
    x="ds", y="y",
    labels={"ds": "Date", "y": "Revenue (£)"},
    color_discrete_sequence=["#1f77b4"]
)
fig.update_layout(
    height=350,
    margin=dict(l=0, r=0, t=30, b=0),
    hovermode="x unified"
)
st.plotly_chart(fig, use_container_width=True)

st.divider()

# Navigation guide
st.subheader("🗺️ Dashboard Navigation")
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.info("**📈 Forecasting**\n\nProphet + LSTM + Hybrid model predictions vs actual revenue")

with col2:
    st.info("**👥 Segmentation**\n\nKMeans + DBSCAN customer clusters with RFM analysis")

with col3:
    st.info("**⚠️ Churn Analysis**\n\nAt-risk customers identified with XGBoost + SHAP")

with col4:
    st.info("**📦 Inventory**\n\nNext 30-day demand forecast with safety stock recommendations")

st.divider()

# Model performance summary
st.subheader("🤖 Model Performance Summary")
col1, col2 = st.columns(2)

with col1:
    st.markdown("**Forecasting Models**")
    perf_df = pd.DataFrame({
        "Model"  : ["Prophet", "LSTM", "Hybrid"],
        "MAE (£)": [12896, 14771, 12780],
        "MAPE %" : [26.47,30.32, 25.80]
    })
    st.dataframe(perf_df, use_container_width=True, hide_index=True)

with col2:
    st.markdown("**Churn Prediction Model**")
    churn_perf = pd.DataFrame({
        "Metric"   : ["AUC-ROC", "Accuracy", "Precision", "Recall"],
        "Value"    : ["0.8034", "0.73", "0.75", "0.67"]
    })
    st.dataframe(churn_perf, use_container_width=True, hide_index=True)

# Footer
st.divider()
st.markdown("""
<div style='text-align: center; color: #888; font-size: 0.85rem;'>
    RetailPulse — Built with Python, Prophet, PyTorch, XGBoost & Streamlit<br>
    Zidio Development Internship Project — 2026
</div>
""", unsafe_allow_html=True)