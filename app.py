import streamlit as st
import pandas as pd
import joblib

# ------------------ LOAD ARTIFACTS ------------------
model = joblib.load("loan_model.pkl")
scaler = joblib.load("standard_scaler.pkl")
feature_names = joblib.load("feature_names.pkl")
home_map = joblib.load("person_home_ownership_mapping.pkl")
defaults_map = joblib.load("previous_loan_defaults_on_file_mapping.pkl")

THRESHOLD = 0.20

# ------------------ PREPROCESS FUNCTION ------------------
def preprocess_input(data):
    df = pd.DataFrame([data])

    # Map categorical features
    df["person_home_ownership"] = df["person_home_ownership"].map(home_map)
    df["previous_loan_defaults_on_file"] = df["previous_loan_defaults_on_file"].map(defaults_map)

    # Numerical columns
    num_cols = [
        'person_age', 'person_income', 'person_emp_exp', 'loan_amnt',
        'loan_int_rate', 'loan_percent_income',
        'cb_person_cred_hist_length', 'credit_score'
    ]

    df[num_cols] = scaler.transform(df[num_cols])

    return df[feature_names]

# ------------------ STREAMLIT UI ------------------
st.set_page_config(page_title="Loan Default Prediction", layout="centered")

st.title("🏦 Loan Default Prediction System")
st.write("Enter applicant details to assess default risk.")

# ------------------ INPUTS ------------------
age = st.number_input("Age", 18, 75, 30)
income = st.number_input("Annual Income", 5000, 500000, 40000)
emp_exp = st.number_input("Employment Experience (Years)", 0, 40, 5)

home_ownership = st.selectbox(
    "Home Ownership",
    options=list(home_map.keys())
)

loan_amount = st.number_input("Loan Amount", 1000, 500000, 15000)
interest_rate = st.slider("Loan Interest Rate (%)", 5.0, 30.0, 15.0)

loan_percent_income = loan_amount / income if income > 0 else 0.0

prev_default = st.selectbox(
    "Previous Loan Default?",
    options=list(defaults_map.keys())
)

cred_hist_len = st.number_input("Credit History Length (Years)", 0, 40, 5)
credit_score = st.slider("Credit Score", 300, 850, 650)

# ------------------ PREDICTION ------------------
if st.button("🔍 Predict Default Risk"):
    raw_input = {
        'person_age': age,
        'person_income': income,
        'person_emp_exp': emp_exp,
        'person_home_ownership': home_ownership,
        'loan_amnt': loan_amount,
        'loan_int_rate': interest_rate,
        'loan_percent_income': loan_percent_income,
        'previous_loan_defaults_on_file': prev_default,
        'cb_person_cred_hist_length': cred_hist_len,
        'credit_score': credit_score
    }

    processed = preprocess_input(raw_input)
    prob_default = model.predict_proba(processed)[0][1]

    st.subheader("📊 Result")

    st.write(f"**Probability of Default:** `{prob_default:.2%}`")

    if prob_default >= THRESHOLD:
        st.error("❌ HIGH RISK — Loan Likely to Default")
    else:
        st.success("✅ LOW RISK — Loan Likely Safe")

    st.caption(f"Decision Threshold: {THRESHOLD:.0%}")
