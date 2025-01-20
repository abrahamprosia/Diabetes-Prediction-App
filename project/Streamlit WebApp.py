import streamlit as st
import joblib
import numpy as np
from scipy.stats import norm


# Load the saved models and scaler
logreg_model = joblib.load("logistic_regression_model.pkl")
knn_model = joblib.load("KNN_model.pkl")
svm_model = joblib.load("svm_model.pkl")
rf_model = joblib.load("random_forest_model.pkl")
scaler = joblib.load("scaler.pkl")

# Function to calculate margin of error
def calculate_confidence(probability, confidence_level=0.99):
    z_score = norm.ppf((1 + confidence_level) / 2)
    margin_of_error = z_score * np.sqrt(probability * (1 - probability))
    return margin_of_error

# Making the web app
st.markdown("""
    <style>
        .title-container {
            background-color: #FA8072;  /* Red background for the title */
            padding: 10 px 0;  /* Adjust padding to control vertical space */
            text-align: center;
            border-radius: 10px;
            
        }
        .rectangle {
            height: 20px;
            width: 100%;
            margin-bottom: 3px;
        }
    </style>
    <div class="title-container">
        <h2>Diabetes Prediction App</h2>
    </div>
    <div class="rectangle"></div>
""", unsafe_allow_html=True)

st.markdown(
    """ <p style = "font-size:15px;
            text-align: center;">
    Enter the patient's details</p>
    """, unsafe_allow_html=True
)


# Input fields
gender = st.radio("Gender:", options=["Male", "Female"])
gender_value = 0 if gender == "Male" else 1

age = st.number_input("Age:", min_value=0, max_value=120, step=1)

bmi = st.number_input("BMI:", min_value=0.0, step=0.1)
if bmi > 50:
    st.warning("BMI seems too high; please recheck")

hba1c_level = st.number_input("HbA1c Level:", min_value=0.0, step=0.1)

blood_glucose_level = st.number_input("Blood Glucose Level:", min_value=0.0, step=1.0)

hypertension = st.radio("Does the patient have hypertension?", options=["No", "Yes"])
hypertension_value = 0 if hypertension == "No" else 1

heart_disease = st.radio("Does the patient have heart disease?", options=["No", "Yes"])
heart_disease_value = 0 if heart_disease == "No" else 1

categ_mapping = {
    "No Info": 0,
    "never": 1,
    "current": 2,
    "former": 3,
    "ever": 4,
    "not current": 5
}
smoking_history = st.radio("Smoking History", options=["No Info", "never", "current", "former", "ever", "not current"])
smoking_history_value = categ_mapping[smoking_history]


# Let the user choose the ML model for prediction
with st.sidebar:
    model_choice = st.radio(
        "Choose Prediction Model:",
        ["Logistic Regression", "KNN", "SVM", "Random Forest"]
    )

# Predict button
if st.button("Predict"):
    feature_values = np.array([[gender_value, age, hypertension_value, heart_disease_value,
                                 smoking_history_value, bmi, hba1c_level, blood_glucose_level]])
    scaled_feature_values = scaler.transform(feature_values)

    # Select model based on user choice
    if model_choice == "Logistic Regression":
        model = logreg_model
    elif model_choice == "KNN":
        model = knn_model
    elif model_choice == "SVM":
        model = svm_model
    elif model_choice == "Random Forest":
        model = rf_model

    # Predict and calculate probabilities (if possible)
    prediction = model.predict(scaled_feature_values)
    probabilities = None
    margin_of_error = None

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(scaled_feature_values)[0, 1]  # Probability of being diabetic (class 1)
        margin_of_error = calculate_confidence(probabilities)

    # Display the result
    if prediction[0] == 1:
        st.error("The patient might be diabetic,consider consulting a healthcare provider for further advice.")
    else:
        st.success("The patient is not diabetic.")

    # Display probabilities and margin of error if available
    if probabilities is not None:
        st.markdown(f"<p style= 'font-size:12px; margin-bottom: 2px;'> Probability of being diabetic: {probabilities:.2%}</p>", unsafe_allow_html=True)
        st.markdown(f"<p style= 'font-size: 12px; margin-top: 2px;'> Margin of Error: ±{margin_of_error:.2%}</p>", unsafe_allow_html=True)
    else:
        st.warning("The selected model does not support confidence level estimation.")
