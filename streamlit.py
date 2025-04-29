import streamlit as st
import joblib
import numpy as np
st.set_page_config(page_title="Cancer Risk Prediction App")

st.title("🔬 Cancer Risk Prediction App")
st.markdown("This application predicts the risk of cancer based on patient information.")


# Load model dan scaler
model = joblib.load("model.pkl")        # pastikan file ini tersedia di folder sama
scaler = joblib.load("scaler.pkl")

# Ambil input dari user
age = st.slider("Age", 20, 90, 30)
gender = st.selectbox("Gender", ["Male", "Female"])
blood_pressure = st.slider("Blood Pressure", 80, 200, 120)
cancer_type = st.selectbox("Cancer Type", ["Type A", "Type B", "Type C"])
radiation = st.selectbox("Radiation Therapy", ["Yes", "No"])
history = st.selectbox("Cancer History in Family", ["Yes", "No"])
smoker = st.selectbox("Smoker", ["Yes", "No"])
lifestyle = st.selectbox("Sedentary Lifestyle", ["Yes", "No"])

# Konversi ke angka jika perlu
gender_val = 1 if gender == "Male" else 0
cancer_val = {"Type A": 100, "Type B": 200, "Type C": 300}[cancer_type]
radiation_val = 1 if radiation == "Yes" else 0
history_val = 1 if history == "Yes" else 0
smoker_val = 1 if smoker == "Yes" else 0
lifestyle_val = 1 if lifestyle == "Yes" else 0

input_data = np.array([[age, gender_val, blood_pressure, cancer_val, radiation_val,
                        history_val, smoker_val, lifestyle_val]])

# Proses saat tombol ditekan
if st.button("Predict"):
    scaled_data = scaler.transform(input_data)
    prediction = model.predict(scaled_data)[0]
    result = "Positive" if prediction == 1 else "Negative"
    st.write(f"⚠️ Cancer Risk Prediction: **{result}**")
