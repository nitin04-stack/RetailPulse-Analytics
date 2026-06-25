# 📊 RetailPulse — AI-Powered Customer Analytics & Demand Forecasting

> End-to-End Data Science Platform for Retail Demand Prediction & Customer Insights

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://retailpulse-analytics-nitinn.streamlit.app/)

## 🚀 Live Demo
👉 [RetailPulse Dashboard](https://retailpulse-analytics-nitinn.streamlit.app/)

## 📌 Project Overview
RetailPulse is an end-to-end retail analytics platform built on the 
Online Retail II dataset (UK). It delivers demand forecasting, 
customer segmentation, churn prediction, and inventory optimization.

## 🎯 Key Results
| Model | Metric | Result |
|-------|--------|--------|
| Prophet | MAE | £12,896 |
| LSTM | MAE | £14,771 |
| Hybrid Ensemble | MAE | £12,780 (Best) |
| Churn XGBoost | AUC-ROC | 0.80 |

## 🛠️ Tech Stack
- **Forecasting:** Prophet + LSTM (PyTorch Lightning)
- **Classification:** XGBoost + SHAP
- **Clustering:** KMeans + DBSCAN
- **Dashboard:** Streamlit + Plotly
- **MLOps:** MLflow + Drift Detection
- **Deployment:** Streamlit Cloud

## 📁 Project Structure
RetailPulse/
|── data/
├── app.py                  # Main dashboard
├── pages/                  # Multi-page Streamlit
├── notebooks/              # Analysis notebooks
└── requirements.txt

## ⚙️ Local Setup
```bash
git clone https://github.com/TERA_USERNAME/RetailPulse-Analytics.git
cd RetailPulse-Analytics
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

## 📊 Dataset
- **Source:** UCI Online Retail II Dataset
- **Period:** Dec 2009 — Dec 2011
- **Market:** United Kingdom
- **Transactions:** 700,388 (after cleaning)

## 🏗️ Architecture
Raw Data → Cleaning → EDA → Feature Engineering
↓
RFM Segmentation (KMeans + DBSCAN)
↓
Demand Forecasting (Prophet + LSTM → Hybrid)
↓
Churn Prediction (XGBoost + SHAP)
↓
Inventory Optimization
↓
MLflow Tracking + Drift Detection
↓
Streamlit Dashboard → Streamlit Cloud

## 👨‍💻 Author
**Nitin** — Data Science Intern @ Zidio Development
