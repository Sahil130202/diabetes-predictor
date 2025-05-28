import streamlit as st
import numpy as np
import joblib
from PIL import Image

#Load the model
model = joblib.load("diabetes_model.pkl")

#Page setup
st.set_page_config(page_title="Diabetes Prediction App", page_icon=":hospital:", layout="wide")
st.title("Diabetes Prediction App")
st.markdown("This app predicts whether a patient has diabetes based on various health metrics.")

#Sidebar - User Inputs
st.sidebar.header("Input health Metrics")

pregnancies = st.sidebar.number_input("Pregnancies", min_value=0, max_value=20, value=0)
glucose = st.sidebar.number_input("Glucose Level", min_value=0, max_value=200, value=100)
blood_pressure = st.sidebar.number_input("Blood Pressure (mm Hg)", min_value=0, max_value=200, value=70)
skin_thickness = st.sidebar.number_input("Skin Thickness (mm)", min_value=0, max_value=100, value=20)
insulin = st.sidebar.number_input("Insulin Level (mu U/ml)", min_value=0, max_value=500, value=80)
bmi = st.sidebar.number_input("BMI", min_value=0.0, max_value=50.0, value=25.0)
dpf = st.sidebar.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=2.5, value=0.5)
age = st.sidebar.number_input("Age", min_value=0, max_value=120, value=30)

#Collect input images
features =  np.array([[pregnancies, glucose, blood_pressure, skin_thickness,insulin, bmi, dpf, age]])


#Make prediction
prediction = model.predict(features)[0]

#Display resu;t
st.subheader("Prediction Result")
if prediction == 1:
    st.error("The model predicts that the patient has diabetes.")
else:
    st.success("The model predicts that the patient does not have diabetes.")

#Show model metrics
st.subheader("Model Metrics")
col1, col2 = st.columns(2)
with col1:
    st.image("metrics_plot_accuracy.png", caption="Model Accuracy and ROC AUC",use_column_width=True)
with col2:
    st.image("metrics_plot_roc.png", caption="ROC Curve", use_column_width=True)

#Footer 
st.markdown("---")
st.markdown("Developed by [Sahil Nayak]")
