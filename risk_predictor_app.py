import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from datetime import datetime

# -------------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------------
st.set_page_config(
    page_title="Recongence AI — Credit Risk Engine",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------------------------------------------
# DARK FINTECH CSS (STRIPE-STYLE THEME)
# -------------------------------------------------------
st.markdown("""
<style>

html, body, [class*="css"]  {
    font-family: 'Inter', sans-serif;
}

/* Background */
.stApp {
    background-color: #0e0f11;
    color: #e2e8f0;
}

/* Container spacing */
.block-container {
    padding-top: 2rem;
}

/* Clean title */
.app-title {
    font-size: 30px;
    font-weight: 800;
    color: #ffffff;
    letter-spacing: -0.5px;
}

/* Subtext */
.subtext {
    font-size: 14px;
    color: #94a3b8;
}

/* Card */
.card {
    background: #14161a;
    padding: 22px 25px;
    border-radius: 12px;
    border: 1px solid #1f2329;
    box-shadow: 0 4px 12px rgba(0,0,0,0.45);
    margin-bottom: 22px;
}

/* Input styling */
.stTextInput>div>div>input,
.stNumberInput>div>div>input,
.stSelectbox>div>div,
.stSlider>div>div>div {
    background-color: #1c1f24 !important;
    color: #ffffff !important;
    border-radius: 8px;
}

.stSelectbox label, .stSlider label {
    color: #e2e8f0 !important;
}

/* Button */
.stButton>button {
    background: #2563eb;
    border: none;
    color: white;
    padding: 0.65rem 1.2rem;
    border-radius: 7px;
    font-weight: 600;
    transition: 0.2s ease;
}

.stButton>button:hover {
    background: #1e3fa1;
    transform: scale(1.02);
}

/* PD Gauge bar */
.stProgress > div > div > div {
    background-color: #3b82f6 !important;
}

/* Result boxes */
.result-good {
    background: rgba(16,185,129,0.15);
    border-left: 4px solid #10b981;
    padding: 12px;
    border-radius: 6px;
    margin-top: 12px;
}

.result-bad {
    background: rgba(239,68,68,0.15);
    border-left: 4px solid #ef4444;
    padding: 12px;
    border-radius: 6px;
    margin-top: 12px;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #111214;
    border-right: 1px solid #1f2329;
}

</style>
""", unsafe_allow_html=True)


# -------------------------------------------------------
# LOAD ARTIFACTS
# -------------------------------------------------------
MODEL_FILE = "xgb_risk_model.joblib"
PRE_FILE = "preprocessor.joblib"

@st.cache_resource
def load_artifacts():
    if not Path(MODEL_FILE).exists() or not Path(PRE_FILE).exists():
        return None, None, "missing"
    try:
        model = joblib.load(MODEL_FILE)
        pre = joblib.load(PRE_FILE)
        return model, pre, None
    except Exception as e:
        return None, None, str(e)

model, preprocessor, load_err = load_artifacts()


# -------------------------------------------------------
# HEADER
# -------------------------------------------------------
# HEADER
st.markdown("""
<div style="text-align:center; margin-bottom: 10px;">
    <div class="app-title">Recongence AI</div>
    <div class="subtext">Enterprise Credit Risk Intelligence Engine</div>
</div>
""", unsafe_allow_html=True)

st.markdown("<hr style='border: 0.5px solid #1f2329; margin-bottom: 25px;'>", unsafe_allow_html=True)

# Now main layout starts normally:
left, right = st.columns([1.1, 0.9])


# -------------------------------------------------------
# PREDICTION LOGIC
# -------------------------------------------------------
def make_prediction(data, model, pre):
    df = pd.DataFrame([data])

    df["LTI_Ratio"] = df["LoanAmount"] / df["Income"].replace(0, np.nan)
    df["LTI_Ratio"] = df["LTI_Ratio"].fillna(df["LTI_Ratio"].median())

    df["DTI_MISSING_FLAG"] = df["DTIRatio"].isnull().astype(int)

    if "Age" in df:
        df["Age_Group"] = pd.cut(
            df["Age"],
            bins=[0,25,35,45,55,99],
            labels=["<25","25-35","36-45","46-55",">55"]
        ).astype(str)
        df.drop(columns=["Age"], inplace=True)

    X = pre.transform(df)
    return float(model.predict_proba(X)[:,1][0])


# -------------------------------------------------------
# SIDEBAR
# -------------------------------------------------------
with st.sidebar:
    st.header("Status")
    if load_err is None:
        st.success("Model Loaded")
    else:
        st.error("Artifacts missing")

    st.markdown("### Deployment Tips")
    st.write("- Keep `.joblib` files in same directory.")
    st.write("- Run with `streamlit run risk_predictor_app.py`.")
    st.caption("Recongence AI · 2025")


# -------------------------------------------------------
# MAIN LAYOUT
# -------------------------------------------------------
left, right = st.columns([1.1, 0.9])

# ---------------- LEFT (INPUT FORM)
with left:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Applicant Details")

    with st.form("predict_form"):

        c1, c2 = st.columns(2)
        with c1:
            income = st.number_input("Annual Income ($)", value=75000)
            loan_amt = st.number_input("Loan Amount ($)", value=25000)
            credit = st.slider("Credit Score", 300, 850, 680)
            dti = st.slider("DTI Ratio", 0.0, 1.0, 0.35)

        with c2:
            int_rate = st.number_input("Interest Rate (%)", value=12.5)
            loan_term = st.selectbox("Loan Term", [12,24,36,48,60,84])
            age = st.slider("Age", 18, 90, 35)
            months = st.number_input("Months Employed", value=60)

        st.markdown("---")

        c3, c4 = st.columns(2)
        with c3:
            edu = st.selectbox("Education", ["Bachelor's","Master's","High School","PhD","Other"])
            lines = st.number_input("Number of Credit Lines", value=3)
            purpose = st.selectbox("Loan Purpose", ["Business","Education","Auto","Home Improvement","Other"])

        with c4:
            marital = st.selectbox("Marital Status", ["Married","Single","Divorced"])
            emp = st.selectbox("Employment Type", ["Full-time","Part-time","Self-employed","Unemployed","Retired"])
            mort = st.selectbox("Has Mortgage?", ["Yes","No"])
            cos = st.selectbox("Has Co-Signer?", ["Yes","No"])

        submitted = st.form_submit_button("Predict Risk")

    st.markdown("</div>", unsafe_allow_html=True)

# ---------------- RIGHT (RESULTS)
with right:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Risk Assessment")

    if submitted:
        if model is None:
            st.error("Model not loaded")
        else:
            data = {
                "Age": age,
                "Income": income,
                "LoanAmount": loan_amt,
                "CreditScore": credit,
                "MonthsEmployed": months,
                "NumCreditLines": lines,
                "InterestRate": int_rate,
                "LoanTerm": loan_term,
                "DTIRatio": dti,
                "Education": edu,
                "EmploymentType": emp,
                "MaritalStatus": marital,
                "HasMortgage": mort,
                "HasDependents": "No",
                "LoanPurpose": purpose,
                "HasCoSigner": cos
            }

            with st.spinner("Running model..."):
                pd_val = make_prediction(data, model, preprocessor)

            st.write("### Probability of Default")
            st.progress(min(max(pd_val, 0), 1))

            pd_pct = pd_val * 100

            st.write(f"**PD: {pd_pct:.2f}%**")

            if pd_val > 0.15:
                st.markdown("<div class='result-bad'>High Risk — Manual Review Needed</div>", unsafe_allow_html=True)
            else:
                st.markdown("<div class='result-good'>Low Risk — Eligible</div>", unsafe_allow_html=True)

    else:
        st.write("Fill the form and click Predict.")

    st.markdown("</div>", unsafe_allow_html=True)
