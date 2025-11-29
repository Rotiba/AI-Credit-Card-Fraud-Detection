# -*- coding: utf-8 -*-
"""
Created on Wed Nov 26 22:30:47 2025

@author: HP
"""


import streamlit as st
import pandas as pd
import shap
import joblib

# Load trained model
model = joblib.load("fraud_model.pkl")

# Friendly names for PCA features
friendly_feature_names = {
    "Time": "Transaction timing",
    "Amount": "Transaction amount",
    "V1": "Pattern 1 (unusual spending behaviour)",
    "V2": "Pattern 2 (irregular transaction rhythm)",
    "V3": "Pattern 3 (sudden deviation from normal behaviour)",
    "V4": "Pattern 4 (rare deviation in spending flow)",
    "V5": "Pattern 5 (anomalous usage pattern)",
    "V6": "Pattern 6 (weak anomaly indicator)",
    "V7": "Pattern 7 (moderate behavioural deviation)",
    "V8": "Pattern 8 (suspicious transaction style)",
    "V9": "Pattern 9 (irregular customer activity)",
    "V10": "Pattern 10 (atypical spending signal)",
    "V11": "Pattern 11 (behavioural fluctuation)",
    "V12": "Pattern 12 (change in spending balance)",
    "V13": "Pattern 13 (unusual feature blend)",
    "V14": "Strong anomaly pattern (major deviation)",
    "V15": "Pattern 15 (sudden behavioural shift)",
    "V16": "Pattern 16 (distorted transaction pattern)",
    "V17": "Pattern 17 (weak fraud signal)",
    "V18": "Pattern 18 (rare anomaly)",
    "V19": "Pattern 19 (latent abnormality)",
    "V20": "Pattern 20 (small behaviour change)",
    "V21": "Pattern 21 (hidden unusual pattern)",
    "V22": "Pattern 22 (subtle anomaly)",
    "V23": "Pattern 23 (weak behaviour deviation)",
    "V24": "Pattern 24 (slight spending anomaly)",
    "V25": "Pattern 25 (light irregularity)",
    "V26": "Pattern 26 (rare behaviour noise)",
    "V27": "Pattern 27 (low-level anomaly)",
    "V28": "Pattern 28 (minor unusual pattern)"
}

st.title("🔍 Credit Card Fraud Detection")
st.write("Upload a CSV file to detect fraud and view explanations.")

uploaded_file = st.file_uploader("Upload creditcard.csv", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("### Uploaded Data")
    st.dataframe(df.head())

    # Make predictions
    predictions = model.predict(df)
    probabilities = model.predict_proba(df)[:, 1]

    df["Fraud_Prediction"] = predictions
    df["Fraud_Probability"] = probabilities

    st.write("### Prediction Results")
    st.dataframe(df[["Fraud_Prediction", "Fraud_Probability"]].head())

    # Choose a row to explain
    row_index = st.number_input(
        "Select row index to explain:",
        min_value=0,
        max_value=len(df) - 1,
        value=0,
        step=1
    )

    # SHAP explanation
    explainer = shap.TreeExplainer(model.named_steps["classifier"])
    preprocessed = model.named_steps["preprocessor"].transform(df)
    shap_values = explainer.shap_values(preprocessed)

    row_shap = shap_values[1][row_index]

    # Friendly text explanation
    st.write("### Top Contributors to the Prediction")

    feature_names = df.columns
    contributions = list(zip(feature_names, row_shap))

    # Sort by impact
    contributions = sorted(contributions, key=lambda x: abs(x[1]), reverse=True)

    for fname, val in contributions[:10]:
        st.write(f"**{friendly_feature_names.get(fname, fname)}:** {val:+.4f}")

    # Visual SHAP plot
    st.write("### SHAP Force Plot")
    shap.initjs()
    st_shap = shap.force_plot(
        explainer.expected_value[1],
        shap_values[1][row_index],
        df.iloc[row_index],
        matplotlib=True,
        show=False
    )
    st.pyplot(st_shap)
