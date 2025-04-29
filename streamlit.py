import streamlit as st
import numpy as np
import tensorflow as tf
import pickle

# Load TFLite model
interpreter = tf.lite.Interpreter(model_path="model_cancer_prediction.tflite")
interpreter.allocate_tensors()

# Load Scaler
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# Streamlit UI
st.title("Cancer Prediction App")

age = st.number_input("Age", min_value=0, max_value=120, step=1)
gender = st.selectbox("Gender", ["Male", "Female"])
bmi = st.number_input("BMI", min_value=10.0, max_value=50.0, step=0.1)
smoking = st.selectbox("Smoking", ["No", "Yes"])
genetic_risk = st.selectbox("Genetic Risk", ["No", "Yes"])
physical_activity = st.selectbox("Physical Activity", ["Low", "Moderate", "High"])
alcohol_intake = st.selectbox("Alcohol Intake", ["No", "Yes"])
cancer_history = st.selectbox("Cancer History", ["No", "Yes"])

# Encoding input
input_data = np.array([
    age,
    1 if gender == "Male" else 0,
    bmi,
    1 if smoking == "Yes" else 0,
    1 if genetic_risk == "Yes" else 0,
    {"Low": 0, "Moderate": 1, "High": 2}[physical_activity],
    1 if alcohol_intake == "Yes" else 0,
    1 if cancer_history == "Yes" else 0
]).reshape(1, -1)

# Scale input
input_data_scaled = scaler.transform(input_data)

# Run inference with TFLite
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

interpreter.set_tensor(input_details[0]['index'], input_data_scaled.astype(np.float32))
interpreter.invoke()
prediction = interpreter.get_tensor(output_details[0]['index'])

# Show result
st.subheader("Prediction Result")
if prediction[0][0] > 0.5:
    st.error("Cancer Detected (Positive)")
else:
    st.success("No Cancer Detected (Negative)")
