import streamlit as st
import numpy as np
import pandas as pd
import pickle

st.set_page_config(layout="centered")
st.title("Cancer Prediction App")

# Gunakan kolom agar inputnya bisa ditata dengan rapi di tengah
col1, col2 = st.columns(2)

with col1:
    age = st.slider("Age", 20, 90, 30)
    blood_pressure = st.slider("Blood Pressure", 80, 200, 120)
    cancer_type = st.selectbox("Cancer Type", ["Type A", "Type B", "Type C"])
    radiation_therapy = st.selectbox("Radiation Therapy", ["No", "Yes"])

with col2:
    gender = st.selectbox("Gender", ["Male", "Female"])
    cancer_history_family = st.selectbox("Cancer History in Family", ["No", "Yes"])
    smoker = st.selectbox("Smoker", ["No", "Yes"])
    sedentary = st.selectbox("Sedentary Lifestyle", ["No", "Yes"])

# Mapping input categorical ke angka
gender = 1 if gender == "Male" else 2
cancer_type = {"Type A": 100, "Type B": 110, "Type C": 120}[cancer_type]
radiation_therapy = 0 if radiation_therapy == "No" else 1
cancer_history_family = 0 if cancer_history_family == "No" else 1
smoker = 0 if smoker == "No" else 1
sedentary = 0 if sedentary == "No" else 1

# Susun input ke array
input_data = np.array([[age, gender, blood_pressure, cancer_type,
                        radiation_therapy, cancer_history_family,
                        smoker, sedentary]])

# Tampilkan input user
st.subheader("Data yang Dimasukkan")
st.write(pd.DataFrame(input_data, columns=[
    "Age", "Gender", "Blood_Pressure", "Cancer_Type",
    "Radiation_Therapy", "Cancer_History_in_Family",
    "Smoker", "Sedentary_Lifestyle"
]))

# Load dan transformasi dengan scaler
try:
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    input_scaled = scaler.transform(input_data)

    st.subheader("Data Setelah Scaling")
    st.write(pd.DataFrame(input_scaled))

except FileNotFoundError:
    st.error("File 'scaler.pkl' tidak ditemukan.")
except Exception as e:
    st.error(f"Terjadi error: {e}")
