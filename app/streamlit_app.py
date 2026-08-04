import streamlit as st
import joblib
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler

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

    input_df["NormalizedAmount"] = scaler.transform(
        input_df[["Amount"]]
    )

    input_df = input_df.drop(columns=["Amount"])

    prediction = model.predict(input_df)

    if prediction[0] == 1:
        st.error("🚨 Fraudulent Transaction")
    else:
        st.success("✅ Legitimate Transaction")

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

    df = pd.read_csv(uploaded_file)

    st.subheader("Uploaded Data")

    st.dataframe(df.head())

    if st.button("🚀 Predict Uploaded CSV"):

        prediction_df = df.copy()

        # Remove target column
        if "Class" in prediction_df.columns:
            prediction_df = prediction_df.drop(columns=["Class"])

        # Scale Amount exactly like training
        prediction_df["NormalizedAmount"] = scaler.transform(
            prediction_df[["Amount"]]
        )

        prediction_df = prediction_df.drop(columns=["Amount"])

        predictions = model.predict(prediction_df)

        result_df = df.copy()

        result_df["Prediction"] = predictions

        result_df["Prediction"] = result_df["Prediction"].map({
            0: "✅ Legitimate",
            1: "🚨 Fraud"
        })

        st.subheader("Prediction Results")

        st.dataframe(result_df.head(20))