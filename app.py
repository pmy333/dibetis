import streamlit as st
import pickle
import numpy as np

# Load the model
with open('model (1).pkl', 'rb') as f:
    model = pickle.load(f)

# Page configuration
st.set_page_config(page_title="Diabetes Predictor", layout="centered")

# Custom CSS for a better look
st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stButton>button { width: 100%; border-radius: 5px; height: 3em; background-color: #007bff; color: white; }
    .result-box { padding: 20px; border-radius: 10px; text-align: center; font-weight: bold; font-size: 24px; }
    </style>
    """, unsafe_allow_html=True)

st.title("🏥 Diabetes Health Predictor")
st.write("Enter the patient's clinical metrics below to predict diabetes risk.")

# Organizing inputs into columns for a cleaner UI
col1, col2 = st.columns(2)

with col1:
    pregnancies = st.number_input("Pregnancies", min_value=0, step=1, help="Number of times pregnant")
    glucose = st.number_input("Glucose", min_value=0, help="Plasma glucose concentration")
    blood_pressure = st.number_input("Blood Pressure", min_value=0, help="Diastolic blood pressure (mm Hg)")
    skin_thickness = st.number_input("Skin Thickness", min_value=0, help="Triceps skin fold thickness (mm)")

with col2:
    insulin = st.number_input("Insulin", min_value=0, help="2-Hour serum insulin (mu U/ml)")
    bmi = st.number_input("BMI", min_value=0.0, format="%.1f", help="Body mass index (weight in kg/(height in m)^2)")
    dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, format="%.3f", help="Diabetes pedigree function")
    age = st.number_input("Age", min_value=0, step=1)

st.markdown("---")

if st.button("Analyze Results"):
    # Prepare the input array
    features = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]])
    
    # Prediction
    prediction = model.predict(features)
    probability = model.predict_proba(features)

    # Display Result
    if prediction[0] == 1:
        st.error("### Result: High Risk of Diabetes")
        st.write(f"Confidence Level: {probability[0][1]*100:.2f}%")
    else:
        st.success("### Result: Low Risk of Diabetes")
        st.write(f"Confidence Level: {probability[0][0]*100:.2f}%")

st.info("Disclaimer: This tool is for educational purposes and should not replace professional medical advice.")
