import streamlit as st
import pandas as pd
import numpy as np
import joblib  # untuk load model
from sklearn.preprocessing import StandardScaler

# Load model dan scaler (ubah path sesuai file kamu)
scaler = joblib.load("scaler.pkl")

st.title("Cancer Prediction App")

# === Input dari User ===
col1, col2 = st.columns(2)

with col1:
    age = st.slider("Age", 20, 90, 30)
    blood_pressure = st.slider("Blood Pressure", 80, 200, 120)
    cancer_type = st.selectbox("Cancer Type", ["Type A", "Type B", "Type C"])
    radiation_therapy = st.selectbox("Radiation Therapy", ["No", "Yes"])
    gender = st.selectbox("Gender", ["Male", "Female"])
    cancer_history = st.selectbox("Cancer History in Family", ["No", "Yes"])
    smoker = st.selectbox("Smoker", ["No", "Yes"])
    sedentary = st.selectbox("Sedentary Lifestyle", ["No", "Yes"])

# Mapping input menjadi angka
gender = 1 if gender == "Male" else 2
cancer_type = {"Type A": 100, "Type B": 110, "Type C": 120}[cancer_type]
radiation_therapy = 1 if radiation_therapy == "Yes" else 0
cancer_history = 1 if cancer_history == "Yes" else 0
smoker = 1 if smoker == "Yes" else 0
sedentary = 1 if sedentary == "Yes" else 0

# Buat dataframe untuk prediksi
new_data = pd.DataFrame([[
    age, gender, blood_pressure, cancer_type, radiation_therapy,
    cancer_history, smoker, sedentary
]])

# === Tombol Prediksi ===
if st.button("Predict"):
    # Scaling
    scaled_data = scaler.transform(new_data)

    # Tampilkan hasil
    st.subheader("Prediction Result")
    st.write(f"⚠️ Cancer Risk Prediction: **{'Positive' if prediction == 1 else 'Negative'}**")
