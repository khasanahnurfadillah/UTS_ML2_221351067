import streamlit as st
import numpy as np
import tensorflow as tf
import joblib

# Load scaler
scaler = joblib.load("scaler.pkl")

# Load .tflite model
interpreter = tf.lite.Interpreter(model_path="model_cancer_prediction.tflite")
interpreter.allocate_tensors()

# Get input/output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# App UI
st.title("Cancer Prediction App")
st.markdown("Masukkan nilai dari 3 fitur di bawah ini:")

# Input user
perimeter = st.number_input("Perimeter Mean", value=0.00, format="%.2f")
area = st.number_input("Area Mean", value=0.00, format="%.2f")
smoothness = st.number_input("Smoothness Mean", value=0.00, format="%.5f")

# Prediksi saat tombol ditekan
if st.button("Prediksi"):
    try:
        # Buat array input
        input_data = np.array([[perimeter, area, smoothness]])
        
        # Scaling
        input_scaled = scaler.transform(input_data).astype(np.float32)

        # Set input ke model
        interpreter.set_tensor(input_details[0]['index'], input_scaled)

        # Run inference
        interpreter.invoke()

        # Ambil output prediksi
        prediction = interpreter.get_tensor(output_details[0]['index'])

        # Tampilkan hasil
        result = "Kanker Ganas" if prediction[0][0] > 0.5 else "Kanker Jinak"
        st.success(f"Hasil Prediksi: {result} (Probabilitas: {prediction[0][0]:.2f})")

    except Exception as e:
        st.error(f"Terjadi error: {str(e)}")
