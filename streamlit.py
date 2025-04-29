import streamlit as st
import numpy as np
import pandas as pd
import pickle

# Judul aplikasi
st.title("Cancer Prediction App")

# Sidebar input user
st.sidebar.header("Input dari User")

age = st.sidebar.slider("Age", 20, 90, 30)

gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
gender = 1 if gender == "Male" else 2  # mapping ke angka

blood_pressure = st.sidebar.slider("Blood Pressure", 80, 200, 120)

cancer_type = st.sidebar.selectbox("Cancer Type", ["Type A", "Type B", "Type C"])
cancer_type = {"Type A": 100, "Type B": 110, "Type C": 120}[cancer_type]  # mapping sesuai dataset

radiation_therapy = st.sidebar.selectbox("Radiation Therapy", ["No", "Yes"])
radiation_therapy = 0 if radiation_therapy == "No" else 1

cancer_history_family = st.sidebar.selectbox("Cancer History in Family", ["No", "Yes"])
cancer_history_family = 0 if cancer_history_family == "No" else 1

smoker = st.sidebar.selectbox("Smoker", ["No", "Yes"])
smoker = 0 if smoker == "No" else 1

sedentary = st.sidebar.selectbox("Sedentary Lifestyle", ["No", "Yes"])
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

# Load scaler
try:
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    input_scaled = scaler.transform(input_data)

    # Tampilkan hasil scaling
    st.subheader("Data Setelah Scaling")
    st.write(pd.DataFrame(input_scaled))

    # Kalau ada model, bisa tambahkan prediksi di sini:
    # with open("model.pkl", "rb") as f:
    #     model = pickle.load(f)
    # prediction = model.predict(input_scaled)
    # st.success(f"Hasil Prediksi: {'Positif' if prediction[0] == 1 else 'Negatif'}")

except FileNotFoundError:
    st.error("File 'scaler.pkl' tidak ditemukan. Pastikan file sudah di-upload.")

except Exception as e:
    st.error(f"Terjadi error: {e}")

