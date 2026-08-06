import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

import streamlit as st
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from agents.validation_agent import ValidationAgent
from agents.fraud_agent import FraudAgent
from agents.risk_agent import RiskAgent
from agents.alert_agent import AlertAgent
from agents.learning_agent import LearningAgent

# ----------------------------
# Page Settings
# ----------------------------
st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="wide"
)

# ----------------------------
# Paths
# ----------------------------
BASE_DIR = Path(__file__).parent.parent

MODEL_PATH = BASE_DIR / "models" / "random_forest_model.pkl"
SCALER_PATH = BASE_DIR / "models" / "scaler.pkl"

# ----------------------------
# Load Model
# ----------------------------
model = joblib.load(MODEL_PATH)

# ----------------------------
# Recreate training scaler
# ----------------------------
scaler = joblib.load(SCALER_PATH)

# ----------------------------
# Initialize Agents
# ----------------------------
validation_agent = ValidationAgent()
fraud_agent = FraudAgent()
risk_agent = RiskAgent()
alert_agent = AlertAgent()
learning_agent = LearningAgent()

# ----------------------------
# Title
# ----------------------------
st.title("💳 Credit Card Fraud Detection System")

# ==========================================================
# Manual Prediction
# ==========================================================

st.header("🔍 Manual Prediction")

feature_names = [
    "Time","V1","V2","V3","V4","V5","V6","V7","V8","V9",
    "V10","V11","V12","V13","V14","V15","V16","V17","V18",
    "V19","V20","V21","V22","V23","V24","V25","V26",
    "V27","V28","Amount"
]

values = []

for feature in feature_names:
    values.append(st.number_input(feature, value=0.0))

if st.button("🔍 Predict Fraud"):

    input_df = pd.DataFrame([values], columns=feature_names)

    amount = input_df["Amount"].iloc[0]

    # Scale Amount
    input_df["NormalizedAmount"] = scaler.transform(
    input_df[["Amount"]].values
    )

    input_df = input_df.drop(columns=["Amount"])

    # Validation Agent
    valid, message = validation_agent.validate(input_df)

    if not valid:
        st.error(message)

    else:

        # Fraud Agent
        prediction = fraud_agent.predict(input_df)

        # Risk Agent
        risk = risk_agent.assess_risk(prediction, amount)

        # Alert Agent
        alert = alert_agent.generate_alert(prediction, risk)

        # Learning Agent
        learning_agent.save_transaction(
            amount,
            prediction,
            risk
        )

        st.subheader("🤖 Agentic AI Result")

        st.success(alert["message"])
        st.write("### Risk Level:", alert["risk"])
        st.write("🕒 Time:", alert["time"])

# ==========================================================
# CSV Upload
# ==========================================================

st.markdown("---")

st.header("📂 Upload CSV File")

uploaded_file = st.file_uploader(
    "Choose a CSV file",
    type=["csv"]
)

if uploaded_file is not None:
    
    rows_to_process = st.slider(
        "Rows to Process",
        min_value=10,
        max_value=500,
        value=100,
        step=10
    )

    df = pd.read_csv(uploaded_file)

    st.subheader("Uploaded Data")

    st.dataframe(df.head())

if st.button("🚀 Predict Uploaded CSV"):
    
    progress = st.progress(0)

    prediction_df = df.head(rows_to_process).copy()
    df = df.head(rows_to_process).copy()

    if "Class" in prediction_df.columns:
        prediction_df = prediction_df.drop(columns=["Class"])

    prediction_df = df.head(100).copy()
    df = df.head(100).copy()

    if "Class" in prediction_df.columns:
        prediction_df = prediction_df.drop(columns=["Class"])

    prediction_df["NormalizedAmount"] = scaler.transform(
        prediction_df[["Amount"]].values
    )
    prediction_df = prediction_df.drop(columns=["Amount"])

    results = []

    for index, row in prediction_df.iterrows():

        single_transaction = pd.DataFrame([row])

        amount = df.iloc[index]["Amount"]

        # Validation Agent
        valid, message = validation_agent.validate(single_transaction)

        if not valid:

            results.append({
                "Prediction": "❌ Invalid",
                "Risk": "-",
                "Status": message
            })

            continue

        # Fraud Agent
        prediction = fraud_agent.predict(single_transaction)

        # Risk Agent
        risk = risk_agent.assess_risk(prediction, amount)

        # Alert Agent
        alert = alert_agent.generate_alert(prediction, risk)

        # Learning Agent
        learning_agent.save_transaction(
            amount,
            prediction,
            risk
        )

        results.append({
            "Prediction": "🚨 Fraud" if prediction == 1 else "✅ Legitimate",
            "Risk": risk,
            "Status": alert["message"]
        })

    result_df = df.copy()

    result_df["Prediction"] = [r["Prediction"] for r in results]
    result_df["Risk"] = [r["Risk"] for r in results]
    result_df["Status"] = [r["Status"] for r in results]

    st.subheader("🤖 Agentic AI Prediction Results")

# ==============================
# Dashboard Statistics
# ==============================

total = len(result_df)
fraud = (result_df["Prediction"] == "🚨 Fraud").sum()
legit = (result_df["Prediction"] == "✅ Legitimate").sum()

high_risk = (result_df["Risk"] == "🔴 HIGH").sum()
medium_risk = (result_df["Risk"] == "🟡 MEDIUM").sum()
low_risk = (result_df["Risk"] == "🟢 SAFE").sum()

col1, col2, col3, col4 = st.columns(4)

col1.metric("📄 Total Transactions", total)
col2.metric("🚨 Fraud Detected", fraud)
col3.metric("✅ Legitimate", legit)
col4.metric("🔴 High Risk", high_risk)

st.markdown("---")

st.dataframe(result_df)

csv = result_df.to_csv(index=False).encode("utf-8")

st.download_button(
    label="📥 Download Prediction Report",
    data=csv,
    file_name="fraud_prediction_report.csv",
    mime="text/csv",
)

st.markdown("## 📊 Analytics Dashboard")

col1, col2 = st.columns(2)

with col1:
    prediction_counts = result_df["Prediction"].value_counts()

    fig, ax = plt.subplots(figsize=(5,4))
    ax.bar(prediction_counts.index, prediction_counts.values)
    ax.set_title("Fraud vs Legitimate")
    ax.set_ylabel("Transactions")

    st.pyplot(fig)

with col2:
    risk_counts = result_df["Risk"].value_counts()

    fig2, ax2 = plt.subplots(figsize=(5,4))
    ax2.pie(
        risk_counts.values,
        labels=risk_counts.index,
        autopct="%1.1f%%"
    )
    ax2.set_title("Risk Distribution")

    st.pyplot(fig2)
    
    # ==========================
# Sidebar
# ==========================

st.sidebar.title("💳 Fraud Detection AI")

st.sidebar.markdown("---")

st.sidebar.header("🤖 AI Agents")

st.sidebar.success("✅ Validation Agent")
st.sidebar.success("✅ Fraud Detection Agent")
st.sidebar.success("✅ Risk Assessment Agent")
st.sidebar.success("✅ Alert Agent")
st.sidebar.success("✅ Learning Agent")

st.sidebar.markdown("---")

st.sidebar.header("📊 Model")

st.sidebar.write("Random Forest Classifier")

st.sidebar.markdown("---")

st.sidebar.header("📈 Performance")

st.sidebar.metric("Accuracy", "99.96%")
st.sidebar.metric("Precision", "94%")
st.sidebar.metric("Recall", "82%")
st.sidebar.metric("F1 Score", "87%")

st.sidebar.markdown("---")

st.sidebar.info(
    "Built with Python, Streamlit, Scikit-learn and Agentic AI."
)