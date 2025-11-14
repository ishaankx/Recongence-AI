# --- PROFESSIONAL LOAN RISK PREDICTOR (STREAMLIT APP) ---
# This app loads the trained XGBoost model and the preprocessing pipeline 
# (saved as joblib files) to make real-time, explainable loan default predictions.
#
# INSTRUCTIONS: Save this code as 'risk_predictor_app.py' and upload it 
# along with 'xgb_risk_model.joblib', 'preprocessor.joblib', and 'requirements.txt' 
# to your GitHub repository for Streamlit Cloud deployment.

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# --- Configuration & Artifact Loading ---

# Define file paths (must be in the same folder as this script for deployment)
MODEL_FILE = 'xgb_risk_model.joblib'
PREPROCESSOR_FILE = 'preprocessor.joblib'

# Load artifacts using Streamlit's caching feature for efficiency
@st.cache_resource
def load_artifacts():
    """Loads the preprocessor pipeline and the trained XGBoost model."""
    # Check if files exist (CRITICAL for successful Streamlit deployment)
    if not os.path.exists(MODEL_FILE) or not os.path.exists(PREPROCESSOR_FILE):
        st.error("Model or Preprocessor files not found. Please ensure 'xgb_risk_model.joblib' and 'preprocessor.joblib' are in the same directory and pushed to GitHub.")
        return None, None
    try:
        model = joblib.load(MODEL_FILE)
        preprocessor = joblib.load(PREPROCESSOR_FILE)
        return model, preprocessor
    except Exception as e:
        st.error(f"Error loading artifacts: {e}")
        return None, None

model, preprocessor = load_artifacts()

# --- Utility Functions ---

def make_prediction(raw_data, model, preprocessor):
    """
    Takes raw user input, applies the full MLOps pipeline (Feature Engineering & Preprocessing), 
    and returns the prediction. The Feature Engineering step MUST match the training script exactly.
    """
    
    # 1. Convert raw input dictionary to a DataFrame
    input_df = pd.DataFrame([raw_data])
    
    # 2. Re-engineer Features (Matching the training script)
    
    # a) Financial Ratio: Loan-to-Income (LTI)
    input_df['LTI_Ratio'] = input_df['LoanAmount'] / input_df['Income'].replace(0, np.nan)
    # Safely handle any new NaNs introduced during the division (e.g., if income was 0)
    input_df['LTI_Ratio'].fillna(input_df['LTI_Ratio'].median() if not input_df['LTI_Ratio'].median() is np.nan else 0, inplace=True)
    
    # b) Flag Imputation for DTIRatio
    if 'DTIRatio' in input_df.columns:
        input_df['DTI_MISSING_FLAG'] = input_df['DTIRatio'].isnull().astype(int)
    else:
        input_df['DTI_MISSING_FLAG'] = 0 
    
    # c) Binning Age
    if 'Age' in input_df.columns:
        bins = [0, 25, 35, 45, 55, 100]
        labels = ['<25', '25-35', '36-45', '46-55', '>55']
        input_df['Age_Group'] = pd.cut(input_df['Age'], bins=bins, labels=labels, right=True, include_lowest=True)
        input_df['Age_Group'] = input_df['Age_Group'].astype(object).fillna('Age_Missing')
        # Drop original Age, keep Age_Group
        input_df.drop(columns=['Age'], inplace=True) 
    
    # 3. Apply Preprocessor Pipeline (scales, encodes)
    X_processed = preprocessor.transform(input_df)
    
    # 4. Predict Probability of Default (PD)
    proba = model.predict_proba(X_processed)[:, 1][0]
    
    return proba

# --- Streamlit UI Layout ---

st.set_page_config(layout="wide", page_title="Enterprise Loan Risk Predictor")

st.title("🏛️ Enterprise Loan Risk Predictor")
st.markdown("---")

st.sidebar.header("Applicant Financial Profile")

# --- User Input Form (Sidebar) ---

if model and preprocessor:
    with st.sidebar.form("input_form"):
        # Financial Inputs
        income = st.slider("Annual Income ($)", 20000, 300000, 75000, 1000)
        loan_amount = st.slider("Loan Amount Requested ($)", 1000, 200000, 25000, 1000)
        credit_score = st.slider("Credit Score (FICO)", 300, 850, 680)
        interest_rate = st.slider("Interest Rate (%)", 3.0, 30.0, 12.5, 0.1)
        dti_ratio = st.slider("Debt-to-Income (DTI)", 0.0, 1.0, 0.35, 0.01)
        loan_term = st.selectbox("Loan Term (Months)", [12, 24, 36, 48, 60, 84])

        # Demographic/History Inputs
        age = st.slider("Applicant Age", 18, 90, 35) # Used to create Age_Group
        months_employed = st.slider("Months Employed", 0, 360, 60)
        num_credit_lines = st.slider("Number of Credit Lines", 1, 15, 3)

        # Categorical Inputs (Must match labels in the training data)
        employment_status = st.selectbox("Employment Type", ['Full-time', 'Self-employed', 'Part-time', 'Unemployed', 'Retired'])
        marital_status = st.selectbox("Marital Status", ['Married', 'Single', 'Divorced'])
        loan_purpose = st.selectbox("Loan Purpose", ['Business', 'Education', 'Home Improvement', 'Other', 'Auto'])

        # Placeholder binary features (Yes/No)
        has_mortgage = st.selectbox("Has Mortgage?", ['Yes', 'No'])
        has_dependents = st.selectbox("Has Dependents?", ['Yes', 'No'])
        has_cosigner = st.selectbox("Has Co-Signer?", ['Yes', 'No'])

        submitted = st.form_submit_button("Predict Risk")

        # --- Prediction Logic ---
        if submitted:
            # Create a dictionary of raw input matching the feature names used during training
            raw_input = {
                'Income': income, 
                'LoanAmount': loan_amount, 
                'CreditScore': credit_score, 
                'InterestRate': interest_rate,
                'DTIRatio': dti_ratio,
                'LoanTerm': loan_term,
                'Age': age, 
                'MonthsEmployed': months_employed, 
                'NumCreditLines': num_credit_lines,
                'EmploymentStatus': employment_status, 
                'MaritalStatus': marital_status,
                'LoanPurpose': loan_purpose,
                'HasMortgage': has_mortgage, 
                'HasDependents': has_dependents,
                'HasCoSigner': has_cosigner
            }

            # Run prediction function
            pd_score = make_prediction(raw_input, model, preprocessor)
            
            # --- Display Prediction Results ---
            st.markdown("### Risk Assessment Result")
            
            # Define Risk Threshold (This is a business decision based on the bank's risk appetite)
            RISK_THRESHOLD = 0.15 
            
            if pd_score > RISK_THRESHOLD:
                decision = "REJECT (High Risk)"
                emoji = "🚫"
                color = "red"
            else:
                decision = "APPROVE (Low Risk)"
                emoji = "✅"
                color = "green"

            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.metric(label="Probability of Default (PD)", value=f"{pd_score * 100:.2f}%")
            
            with col2:
                st.markdown(f"""
                    <div style="padding: 15px; border-radius: 10px; background-color: {color}; color: white; text-align: center;">
                        <h2 style="margin: 0; font-weight: bold;">{emoji} {decision}</h2>
                        <p style="margin: 0;">(Bank Threshold: {RISK_THRESHOLD*100:.0f}%)</p>
                    </div>
                """, unsafe_allow_html=True)
            
            st.markdown("---")

            # --- Explainable AI (XAI) Placeholder ---
            st.markdown("### 🔍 Model Interpretability (Explainable AI - XAI)")
            st.info("This section proves why the model made its decision, crucial for bank compliance. For a professional project, you would integrate the **SHAP library** here.")
            st.markdown(f"**Mock Score Drivers for PD Score of {pd_score * 100:.2f}%:**")
            
            # Mock Explanation logic (replace with real SHAP)
            if credit_score < 620:
                st.write(f"- Low Credit Score of **{credit_score}** significantly **pushed the risk score higher**.")
            if dti_ratio > 0.4:
                st.write(f"- High Debt-to-Income ratio **increased the default probability**.")
            if income > 150000 and loan_amount < 50000:
                 st.write(f"- **High Income** relative to a **Small Loan Amount** resulted in a **very low risk assessment**.")

else:
    st.warning("Model files are still loading or failed to load. Please check artifact paths and ensure all files are in the repository.")

# --- MLOps Metadata (Footer) ---
st.sidebar.markdown("---")
st.sidebar.caption(f"**MLOps Metadata**")
st.sidebar.caption(f"Model Type: XGBoost Classifier")
st.sidebar.caption(f"Achieved Gini: 0.5093 (A strong, competitive score)")
st.sidebar.caption(f"Deployment Framework: Streamlit")