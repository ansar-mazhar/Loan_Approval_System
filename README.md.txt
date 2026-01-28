# Loan Default Prediction System

This project is a machine learning–based loan default prediction system
built using Random Forest and deployed with Streamlit.

## Features Used
- Income & employment
- Loan amount & interest rate
- Debt-to-income ratio
- Credit score & credit history
- Previous loan default history

## Decision Logic
- Model outputs probability of default
- Threshold = 0.20
- Probability ≥ 0.20 → High Risk (Reject)

## How to Run Locally
```bash
pip install -r requirements.txt
streamlit run app.py
