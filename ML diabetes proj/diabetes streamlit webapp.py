import streamlit as st
import joblib
import numpy as np

# Load the saved models and scaler
logreg_model = joblib.load("logistic_regression_model.pkl")
knn_model = joblib.load("knn_model.pkl")
svm_model = joblib.load("svm_model.pkl")
rf_model = joblib.load("random_forest_model.pkl")
scaler = joblib.load("scaler.pkl")

#making the web app

st.title("Diabetes Prediction App")
st.write("Enter the patient's details")

#Input fields where user will input their info
gender = st.radio("Gender:", options= ["Male", "Female"])
gender_value = 0 if gender == "Male" else 1

age = st.slider("Age:", min_value=0, max_value=120, step=1)

hypertension = st.radio("Does the patient have hypertension?", options=["No", "Yes"])
hypertension_value = 0 if hypertension == "No" else 1

heart_disease = st.radio("Does the patient have heart disease?", options=["No", "Yes"])
heart_disease_value = 0 if heart_disease == "No" else 1

bmi = st.number_input("BMI:", min_value=0.0, step=0.1)

hba1c_level = st.number_input("HbA1c Level:", min_value=0.0, step=0.1)

blood_glucose_level = st.number_input("Blood Glucose Level:", min_value=0.0, step=1.0)

#Let the user choose the ML model for prediction
model_choice = st.selectbox(
    "Models:",
    options = ["Logistic Regression", "KNN", "SVM", "Random Forest"]
)

#Predict button
if st.button("Predict"):
    feature_values = np.array([[gender_value, age, hypertension_value, heart_disease_value, bmi, hba1c_level, blood_glucose_level ]])
    scaled_feature_values = scaler.transform(feature_values)

#Select model based on the user choice

    if model_choice == "Logistic Regression":
        prediction = logreg_model.predict(scaled_feature_values)
    elif model_choice == "KNN":
        prediction = KNN_model.predict(scaled_feature_values)
    elif model_choice == "SVM":
        prediction = svm_model.predict(scaled_feature_values)
    elif model_choice == "Random Forest":
        prediction = rf_model.predict(scaled_feature_values)

    #Display the Result
    if prediction[0] == 1:
        st.error(f" The patient might be diabetic")
    else:
        st.success(f"The patient is not diabetic")