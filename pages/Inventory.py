import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

st.set_page_config(page_title="Inventory", page_icon="📦", layout="wide")

st.title("📦 Inventory Optimization")
st.markdown("Prophet-based demand forecast — next 30 days reorder recommendations")

@st.cache_data
def load_inventory():
    return pd.read_csv("data/inventory_plan.csv", parse_dates=["Date"])

inventory = load_inventory()

st.divider()

# KPIs
col1, col2, col3 = st.columns(3)
with col1:
    st.metric(
        "Total Forecasted Revenue",
        f"£{inventory['Forecasted_Revenue'].sum():,.0f}",
        "Next 30 days"
    )
with col2:
    st.metric(
        "Total Safety Stock",
        f"£{inventory['Safety_Stock'].sum():,.0f}",
        "20% buffer"
    )
with col3:
    st.metric(
        "Total Reorder Quantity",
        f"£{inventory['Reorder_Quantity'].sum():,.0f}",
        "Revenue + Safety"
    )

st.divider()

# Inventory chart
st.subheader("📊 30-Day Inventory Plan")
fig = go.Figure()

fig.add_trace(go.Bar(
    x=inventory["Date"],
    y=inventory["Forecasted_Revenue"],
    name="Forecasted Revenue",
    marker_color="#3498db",
    opacity=0.7
))

fig.add_trace(go.Scatter(
    x=inventory["Date"],
    y=inventory["Reorder_Quantity"],
    name="Reorder Quantity (with Safety Stock)",
    line=dict(color="#e74c3c", width=2.5),
    mode="lines+markers",
    marker=dict(size=7)
))

fig.update_layout(
    height=450,
    xaxis_title="Date",
    yaxis_title="Revenue / Stock (£)",
    hovermode="x unified",
    legend=dict(orientation="h", yanchor="bottom", y=1.02),
    margin=dict(l=0, r=0, t=50, b=0)
)
st.plotly_chart(fig, use_container_width=True)

st.divider()

# Table
st.subheader("📋 Daily Inventory Plan")
st.dataframe(
    inventory.style.format({
        "Forecasted_Revenue": "£{:,.2f}",
        "Safety_Stock"      : "£{:,.2f}",
        "Reorder_Quantity"  : "£{:,.2f}"
    }),
    use_container_width=True,
    hide_index=True
)

# Download button
csv = inventory.to_csv(index=False).encode("utf-8")
st.download_button(
    label="⬇️ Download Inventory Plan (CSV)",
    data=csv,
    file_name="inventory_plan_30days.csv",
    mime="text/csv"
)