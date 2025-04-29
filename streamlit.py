import streamlit as st
import numpy as np
import tensorflow as tf
import pickle

# Load model TFLite
interpreter = tf.lite.Interpreter(model_path="model_cancer_prediction.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Load scaler
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

st.title("Cancer Prediction App")

st.write("Masukkan data fitur berikut untuk prediksi kanker.")

# Form Input (ganti sesuai jumlah fitur datasetmu)
radius_mean = st.number_input('Radius Mean', value=0.0)
texture_mean = st.number_input('Texture Mean', value=0.0)
perimeter_mean = st.number_input('Perimeter Mean', value=0.0)
area_mean = st.number_input('Area Mean', value=0.0)
smoothness_mean = st.number_input('Smoothness Mean', value=0.0)

# Buat input data
input_data = np.array([[radius_mean, texture_mean, perimeter_mean, area_mean, smoothness_mean]])

# Scaling input
input_data_scaled = scaler.transform(input_data)

# Prediction
if st.button('Predict'):
    interpreter.set_tensor(input_details[0]['index'], input_data_scaled.astype(np.float32))
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    prediction = np.argmax(output_data)
    
    if prediction == 0:
        st.error("Prediksi: Malignant (Kanker Ganas)")
    else:
        st.success("Prediksi: Benign (Kanker Jinak)")