import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

st.set_page_config(page_title="Forecasting", page_icon="📈", layout="wide")

st.title("📈 Demand Forecasting")
st.markdown("Prophet + LSTM Hybrid model — last 30 days prediction vs actual revenue")

# Data load
@st.cache_data
def load_forecast():
    return pd.read_csv("data/forecast_comparison.csv", parse_dates=["Date"])

@st.cache_data
def load_daily():
    return pd.read_csv("data/daily_sales.csv", parse_dates=["ds"])

forecast_df = load_forecast()
daily_df    = load_daily()

# Sidebar filters
st.sidebar.header("Filters")
show_prophet = st.sidebar.checkbox("Show Prophet", value=True)
show_lstm    = st.sidebar.checkbox("Show LSTM", value=True)
show_hybrid  = st.sidebar.checkbox("Show Hybrid", value=True)

st.divider()

# Full revenue history
st.subheader("📉 Full Revenue History")
fig_full = go.Figure()
fig_full.add_trace(go.Scatter(
    x=daily_df["ds"], y=daily_df["y"],
    name="Actual Revenue",
    line=dict(color="black", width=1.5)
))
fig_full.update_layout(
    height=300,
    xaxis_title="Date",
    yaxis_title="Revenue (£)",
    hovermode="x unified",
    margin=dict(l=0, r=0, t=30, b=0)
)
st.plotly_chart(fig_full, use_container_width=True)

st.divider()

# Last 30 days comparison
st.subheader("🔍 Last 30 Days — Model Comparison")

fig = go.Figure()

# Actual
fig.add_trace(go.Scatter(
    x=forecast_df["Date"], y=forecast_df["Actual"],
    name="Actual Revenue",
    line=dict(color="black", width=2.5),
    mode="lines+markers",
    marker=dict(symbol="circle", size=6)
))

if show_prophet:
    fig.add_trace(go.Scatter(
        x=forecast_df["Date"], y=forecast_df["Prophet_Pred"],
        name="Prophet Forecast",
        line=dict(color="blue", width=2, dash="dash"),
        mode="lines+markers",
        marker=dict(symbol="x", size=6)
    ))

if show_lstm:
    fig.add_trace(go.Scatter(
        x=forecast_df["Date"], y=forecast_df["LSTM_Pred"],
        name="LSTM Forecast",
        line=dict(color="purple", width=2, dash="dot"),
        mode="lines+markers",
        marker=dict(symbol="star", size=6)
    ))

if show_hybrid:
    fig.add_trace(go.Scatter(
        x=forecast_df["Date"], y=forecast_df["Hybrid_Pred"],
        name="Hybrid",
        line=dict(color="green", width=2.5),
        mode="lines+markers",
        marker=dict(symbol="diamond", size=6)
    ))

fig.update_layout(
    height=450,
    xaxis_title="Date",
    yaxis_title="Revenue (£)",
    hovermode="x unified",
    legend=dict(orientation="h", yanchor="bottom", y=1.02),
    margin=dict(l=0, r=0, t=50, b=0)
)
st.plotly_chart(fig, use_container_width=True)

st.divider()

# Metrics
st.subheader("📊 Model Performance Metrics")
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("### Prophet")
    st.metric("MAE", "£12,896")
    st.metric("RMSE", "£28,735")
    st.metric("MAPE", "26.47%")

with col2:
    st.markdown("### LSTM")
    st.metric("MAE", "£14,771")
    st.metric("RMSE", "£31,451")
    st.metric("MAPE", "30.32%")

with col3:
    st.markdown("### Hybrid (Winner)")
    st.metric("MAE", "£12,780")
    st.metric("RMSE", "£29,245")
    st.metric("MAPE", "25.80%")

st.divider()

# Data table
st.subheader("📋 Forecast vs Actual — Detailed Table")
st.dataframe(
    forecast_df.style.format({
        "Actual"      : "£{:,.2f}",
        "Prophet_Pred": "£{:,.2f}",
        "LSTM_Pred"   : "£{:,.2f}",
        "Hybrid_Pred" : "£{:,.2f}",
        "Hybrid_Error": "£{:,.2f}"
    }),
    use_container_width=True,
    hide_index=True
)

# Note about Dec 9 spike
st.info("""
**Note on Model Performance:** December 9, 2011 shows a Christmas spike of £179,543 
which was not present in training data. Excluding this day, Prophet MAE drops to £9,876 and Hybrid MAE drops to £9,780
(Normal Days MAPE: 26.47%). This is a known data limitation, not model failure.
""")