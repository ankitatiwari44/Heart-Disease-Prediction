import streamlit as st
import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler

# Load trained model and scaler
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

st.title("Heart Disease Prediction App")
st.write("Enter the patient's health details below:")

# Input fields
age = st.number_input("Age", 20, 100)
sex = st.selectbox("Sex", [0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
cp = st.selectbox("Chest Pain Type (1–4)", [1, 2, 3, 4])
trestbps = st.number_input("Resting Blood Pressure (trestbps)", 80, 200)
chol = st.number_input("Serum Cholesterol (chol)", 100, 600)
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl (fbs)", [0, 1])
restecg = st.selectbox("Resting ECG (0 = Normal, 1 = ST-T abnormality, 2 = LV hypertrophy)", [0, 1, 2])
thalach = st.number_input("Max Heart Rate Achieved (thalach)", 60, 250)
exang = st.selectbox("Exercise Induced Angina (exang)", [0, 1])
oldpeak = st.number_input("ST Depression (oldpeak)", 0.0, 10.0, step=0.1)
slope = st.selectbox("Slope of ST Segment (1 = Upsloping, 2 = Flat, 3 = Downsloping)", [1, 2, 3])
ca = st.selectbox("Number of Major Vessels (ca)", [0, 1, 2, 3])
thal = st.selectbox("Thalassemia (3 = Normal, 6 = Fixed defect, 7 = Reversible defect)", [3, 6, 7])

# Input dict with dummy variables (one-hot encoding format used during training)
input_dict = {
    'age': age,
    'sex': sex,
    'trestbps': trestbps,
    'chol': chol,
    'fbs': fbs,
    'thalach': thalach,
    'exang': exang,
    'oldpeak': oldpeak,
    'ca': ca,
    'cp_2.0': 0,
    'cp_3.0': 0,
    'cp_4.0': 0,
    'restecg_1.0': 0,
    'restecg_2.0': 0,
    'slope_2.0': 0,
    'slope_3.0': 0,
    'thal_6.0': 0,
    'thal_7.0': 0
}

# Set the one-hot encoded values
if cp in [2, 3, 4]:
    input_dict[f'cp_{float(cp)}'] = 1
if restecg in [1, 2]:
    input_dict[f'restecg_{float(restecg)}'] = 1
if slope in [2, 3]:
    input_dict[f'slope_{float(slope)}'] = 1
if thal in [6, 7]:
    input_dict[f'thal_{float(thal)}'] = 1

# Convert to DataFrame
input_df = pd.DataFrame([input_dict])

# Scale and predict
if st.button("Predict"):
    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)

    st.subheader("Prediction:")
    if prediction[0] == 1:
        st.error(" The patient is likely to have heart disease.")
    else:
        st.success(" The patient is unlikely to have heart disease.")
