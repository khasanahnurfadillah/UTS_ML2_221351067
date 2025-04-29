import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load model dan scaler
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# Judul
st.title("Cancer Prediction App")

# Sidebar untuk input
st.sidebar.header("Masukkan Data Pasien")

# Input dari user
gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
age = st.sidebar.slider("Age", 20, 90, 30)
bmi = st.sidebar.slider("BMI", 15.0, 50.0, 22.5)
smoking = st.sidebar.selectbox("Smoking Status", ["No", "Yes"])
alcohol = st.sidebar.selectbox("Alcohol Intake", ["No", "Yes"])
physical = st.sidebar.selectbox("Physical Activity", ["Low", "Moderate", "High"])
cancer_history = st.sidebar.selectbox("Cancer History", ["No", "Yes"])

# Konversi input ke numerik sesuai pelatihan model
gender = 1 if gender == "Male" else 0
smoking = 1 if smoking == "Yes" else 0
alcohol = 1 if alcohol == "Yes" else 0
cancer_history = 1 if cancer_history == "Yes" else 0
physical_dict = {"Low": 0, "Moderate": 1, "High": 2}
physical = physical_dict[physical]

# Gabungkan input
input_data = np.array([[gender, age, bmi, smoking, alcohol, physical, cancer_history]])

# Scaling
input_scaled = scaler.transform(input_data)

# Tombol prediksi
if st.button("Predict"):
    prediction = model.predict(input_scaled)
    hasil = "Cancer Detected" if prediction[0] == 1 else "No Cancer Detected"
    st.success(f"Prediction Result: {hasil}")
