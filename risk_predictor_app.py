# risk_predictor_app.py (updated - Recongence AI UI)
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from pathlib import Path
from datetime import datetime

# ---------------------------
# Config & Artifacts
# ---------------------------
st.set_page_config(
    page_title="Recongence AI — Enterprise Risk Predictor",
    layout="wide",
    initial_sidebar_state="expanded"
)

MODEL_FILE = "xgb_risk_model.joblib"
PREPROCESSOR_FILE = "preprocessor.joblib"

@st.cache_resource
def load_artifacts():
    if not Path(MODEL_FILE).exists() or not Path(PREPROCESSOR_FILE).exists():
        return None, None, "missing_artifacts"
    try:
        model = joblib.load(MODEL_FILE)
        preprocessor = joblib.load(PREPROCESSOR_FILE)
        return model, preprocessor, None
    except Exception as e:
        return None, None, str(e)

model, preprocessor, load_err = load_artifacts()

# ---------------------------
# Custom CSS (modern card UI)
# ---------------------------
st.markdown(
    """
    <style>
    /* Page background + font sizing */
    .reportview-container .main .block-container { padding-top: 1.2rem; padding-left: 2rem; padding-right:2rem; }
    .recon-header { display:flex; align-items:center; gap:16px; }
    .recon-logo {
        width:56px; height:56px; border-radius:12px;
        background: linear-gradient(135deg,#0f172a,#0ea5a4);
        display:flex; align-items:center; justify-content:center;
        color:white; font-weight:800; font-size:20px;
        box-shadow: 0 6px 18px rgba(2,6,23,0.4);
    }
    .card {
        border-radius:12px; padding:18px; background-color: white;
        box-shadow: 0 6px 20px rgba(15,23,42,0.06); margin-bottom:18px;
    }
    .muted { color: #6b7280; font-size:0.94rem; }
    .big-number { font-size:32px; font-weight:700; }
    .status-strip { height:10px; border-radius:6px; margin-top:8px; }
    .small { font-size:0.86rem; color:#6b7280; }
    .input-label { font-weight:600; }
    .center { text-align:center; }
    /* make container background slightly off-white */
    .stApp { background-color: #f7f9fb; }
    </style>
    """,
    unsafe_allow_html=True
)

# ---------------------------
# Utility: prediction pipeline (keeps your feature engineering)
# ---------------------------
def make_prediction(raw_data, model, preprocessor):
    """
    Accepts raw_data dict -> performs re-engineering -> runs preprocessor -> model.predict_proba
    This mirrors your original pipeline with LTI_Ratio, Age_Group, DTI flag, etc.
    """
    input_df = pd.DataFrame([raw_data])

    # LTI ratio
    input_df['LTI_Ratio'] = input_df['LoanAmount'] / input_df['Income'].replace(0, np.nan)
    input_df['LTI_Ratio'].fillna(input_df['LTI_Ratio'].median() if not np.isnan(input_df['LTI_Ratio'].median()) else 0, inplace=True)

    # DTI missing flag
    input_df['DTI_MISSING_FLAG'] = input_df.get('DTIRatio', pd.Series([np.nan])).isnull().astype(int)

    # Age binning & drop original Age
    if 'Age' in input_df.columns:
        bins = [0,25,35,45,55,100]
        labels = ['<25','25-35','36-45','46-55','>55']
        input_df['Age_Group'] = pd.cut(input_df['Age'], bins=bins, labels=labels, right=True, include_lowest=True)
        input_df['Age_Group'] = input_df['Age_Group'].astype(object).fillna('Age_Missing')
        input_df.drop(columns=['Age'], inplace=True)

    # Ensure columns/types match training-time expectations before preprocessor
    # (The preprocessor should handle encoding and missing value replacements)
    X_processed = preprocessor.transform(input_df)
    proba = model.predict_proba(X_processed)[:,1][0]
    return float(proba)

# ---------------------------
# Header
# ---------------------------
with st.container():
    col1, col2 = st.columns([6, 4])
    with col1:
        st.markdown(
            """
            <div class="recon-header">
                <div class="recon-logo">RA</div>
                <div>
                    <div style="font-size:20px; font-weight:800">Recongence AI</div>
                    <div class="muted">Enterprise loan default & credit risk predictor</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown("<div class='center small'>Model status & run info</div>", unsafe_allow_html=True)
        now = datetime.now().strftime("%b %d, %Y — %H:%M")
        if load_err is None:
            st.success("Model & preprocessor loaded ✓")
            st.caption(f"Loaded at: {now}")
        elif load_err == "missing_artifacts":
            st.error("Model artifacts missing. Put xgb_risk_model.joblib & preprocessor.joblib in app folder.")
        else:
            st.error(f"Artifact load error: {load_err}")

st.markdown("")  # spacing

# ---------------------------
# Main layout: Inputs + Results
# ---------------------------
left_col, right_col = st.columns([1.1, 0.95])

# Input card
with left_col:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Applicant Financial Profile")
    st.write("<div class='small muted'>Fill applicant details — recommended defaults provided for quick testing.</div>", unsafe_allow_html=True)

    with st.form("predict_form", clear_on_submit=False):
        # Financial row
        f1, f2 = st.columns(2)
        with f1:
            income = st.number_input("Annual Income ($)", min_value=0, value=75000, step=5000, format="%d")
            loan_amount = st.number_input("Loan Amount Requested ($)", min_value=0, value=25000, step=1000, format="%d")
            credit_score = st.slider("Credit Score (FICO)", 300, 850, 680)
        with f2:
            interest_rate = st.number_input("Interest Rate (%)", min_value=0.0, value=12.5, step=0.1, format="%.2f")
            dti_ratio = st.slider("Debt-to-Income (DTI)", 0.0, 1.0, 0.35, 0.01)
            loan_term = st.selectbox("Loan Term (Months)", [12,24,36,48,60,84])

        st.markdown("---")
        st.subheader("Demographic & History")
        d1, d2 = st.columns(2)
        with d1:
            age = st.slider("Applicant Age", 18, 90, 35)
            months_employed = st.number_input("Months Employed", min_value=0, value=60, step=1)
            num_credit_lines = st.number_input("Number of Credit Lines", min_value=0, value=3, step=1)
        with d2:
            education = st.selectbox("Education Level", ["Bachelor's","Master's","High School","PhD","Other"])
            marital_status = st.selectbox("Marital Status", ['Married','Single','Divorced'])
            employment_type = st.selectbox("Employment Type", ['Full-time','Self-employed','Part-time','Unemployed','Retired','Other'])

        st.markdown("---")
        st.subheader("Loan Details & Flags")
        l1, l2, l3 = st.columns(3)
        with l1:
            loan_purpose = st.selectbox("Loan Purpose", ['Business','Education','Home Improvement','Auto','Other'])
        with l2:
            has_mortgage = st.selectbox("Has Mortgage?", ['Yes','No'])
            has_dependents = st.selectbox("Has Dependents?", ['Yes','No'])
        with l3:
            has_cosigner = st.selectbox("Has Co-Signer?", ['Yes','No'])
            # optional: allow missing DTIRatio input (use blank to mark missing)
            provide_dti = st.checkbox("Provide precise DTI value?", value=True, help="Uncheck to mark DTI as missing")
            if not provide_dti:
                dti_ratio = np.nan

        submitted = st.form_submit_button("▶ Predict Risk — Run Model")

    st.markdown('</div>', unsafe_allow_html=True)

# Result card
with right_col:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Risk Assessment")

    if submitted:
        if model is None or preprocessor is None:
            st.error("Model or preprocessor not available. Check artifacts in app folder.")
        else:
            # Prepare input dict for pipeline (match training columns)
            raw_input = {
                'Age': age,
                'Income': income,
                'LoanAmount': loan_amount,
                'CreditScore': credit_score,
                'MonthsEmployed': months_employed,
                'NumCreditLines': num_credit_lines,
                'InterestRate': interest_rate,
                'LoanTerm': loan_term,
                'DTIRatio': dti_ratio,
                'Education': education,
                'EmploymentType': employment_type,
                'MaritalStatus': marital_status,
                'HasMortgage': has_mortgage,
                'HasDependents': has_dependents,
                'LoanPurpose': loan_purpose,
                'HasCoSigner': has_cosigner
            }

            # Run prediction
            try:
                with st.spinner("Running model..."):
                    pd_score = make_prediction(raw_input, model, preprocessor)
            except Exception as e:
                st.exception(f"Prediction failed: {e}")
                pd_score = None

            if pd_score is not None:
                PD_pct = pd_score * 100
                # Threshold (business-configurable)
                RISK_THRESHOLD = 0.15

                # Color + decision
                if pd_score > RISK_THRESHOLD:
                    decision = "REJECT — High Risk"
                    status_color = "#ef4444"  # red
                    emoji = "🚫"
                else:
                    decision = "APPROVE — Low Risk"
                    status_color = "#16a34a"  # green
                    emoji = "✅"

                # Big number + progress
                st.markdown(f"<div class='big-number center'>{PD_pct:0.2f}%</div>", unsafe_allow_html=True)
                st.markdown(f"<div class='small center muted'>Probability of Default (PD)</div>", unsafe_allow_html=True)
                st.progress(min(max(pd_score, 0.0), 1.0))

                # Status strip and decision card
                st.markdown(f"""
                    <div style="padding:14px; border-radius:10px; margin-top:12px; background: linear-gradient(90deg, rgba(255,255,255,0.025), rgba(255,255,255,0));">
                        <div style="display:flex; justify-content:space-between; align-items:center;">
                            <div style="font-weight:700;">{emoji} {decision}</div>
                            <div style="font-size:0.9rem; color:#6b7280;">Bank Threshold: {RISK_THRESHOLD*100:.0f}%</div>
                        </div>
                        <div class="status-strip" style="background:{status_color};"></div>
                    </div>
                """, unsafe_allow_html=True)

                # Quick actionable recommendations
                st.markdown("**Suggested next steps:**")
                recs = []
                if pd_score > RISK_THRESHOLD:
                    recs = [
                        "Request additional collateral or guarantor",
                        "Require higher down payment",
                        "Consider shorter loan term or higher interest",
                        "Request latest paystubs / bank statements"
                    ]
                    st.warning("High risk — consider stricter underwriting or manual review.")
                else:
                    recs = [
                        "Fast-track loan approval with standard KYC checks",
                        "Offer pre-approved limits",
                        "Consider competitive interest rate offers"
                    ]
                    st.success("Low risk — candidate likely eligible for standard approval.")
                for r in recs:
                    st.write(f"- {r}")

                # Small XAI mock & integration hint
                st.markdown("---")
                st.markdown("### 🔍 Explainability (XAI)")
                st.info("Enable SHAP or another explainability tool for production. Below is a quick mock summary; replace this with SHAP values for each real input feature.")
                # Mock drivers (replace with real SHAP values)
                st.write("**Top simulated drivers**")
                mock_drivers = []
                if credit_score < 620:
                    mock_drivers.append(("- Low credit score", "increases risk"))
                if not np.isnan(dti_ratio) and dti_ratio > 0.4:
                    mock_drivers.append(("- High DTI", "increases risk"))
                if income > 150000 and loan_amount < 50000:
                    mock_drivers.append(("- High income, small loan", "reduces risk"))

                if mock_drivers:
                    for m,d in mock_drivers:
                        st.write(f"{m} — *{d}*")
                else:
                    st.write("- No single feature dominates; model used a combination of signals.")

                # Celebrate on low-risk or draw attention on high-risk
                if pd_score <= RISK_THRESHOLD:
                    st.balloons()

    else:
        st.markdown("<div class='muted'>Submit the form on the left to produce a risk score. Example inputs are prefilled to help you try it quickly.</div>", unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

# ---------------------------
# Sidebar (metadata & tips)
# ---------------------------
with st.sidebar:
    st.markdown("## Model & App")
    st.write("Recongence AI — Loan Default Predictor")
    if load_err is None:
        st.success("Artifacts loaded")
    elif load_err == "missing_artifacts":
        st.error("Artifacts missing — upload xgb_risk_model.joblib & preprocessor.joblib")
    else:
        st.error("Artifact load error")

    st.markdown("---")
    st.markdown("### Deployment Tips")
    st.write("- Ensure `xgb_risk_model.joblib` and `preprocessor.joblib` are in the same folder.")
    st.write("- Requirements pinned in `requirements.txt` should include streamlit, xgboost, scikit-learn (1.6.1), joblib, pandas, numpy.")
    st.write("- Use `streamlit run risk_predictor_app.py` to start.")
    st.markdown("---")
    st.caption("Recongence AI · MLOps: XGBoost · UI: Streamlit")

# ---------------------------
# Footer
# ---------------------------
st.markdown("<hr>")
st.markdown('<div class="small muted center">Built with ❤️ · Recongence AI · keep model & preprocessor files in the app folder</div>', unsafe_allow_html=True)
