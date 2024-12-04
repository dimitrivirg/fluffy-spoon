
import streamlit as st
import pickle
import numpy as np

# Title and description
st.title("Heart Disease Prediction App")
st.write("This app predicts the likelihood of heart disease based on user input.")

# Load the model
def load_model():
    with open('model.pkl', 'rb') as file:
        model = pickle.load(file)
    return model

model = load_model()

# Input form using widgets
st.header("Enter Patient Information")
with st.form("prediction_form"):
    age = st.number_input("Age", min_value=1, max_value=120, value=50)
    sex = st.radio("Sex", options=[1, 0], format_func=lambda x: "Male" if x == 1 else "Female")
    cp = st.selectbox("Chest Pain Type", options=[0, 1, 2, 3])
    trestbps = st.number_input("Resting Blood Pressure", min_value=50, max_value=200, value=120)
    chol = st.number_input("Serum Cholesterol (mg/dl)", min_value=100, max_value=500, value=200)
    fbs = st.radio("Fasting Blood Sugar > 120 mg/dl", options=[1, 0], format_func=lambda x: "Yes" if x == 1 else "No")
    restecg = st.selectbox("Resting ECG Results", options=[0, 1, 2])
    thalach = st.number_input("Max Heart Rate Achieved", min_value=50, max_value=250, value=150)
    exang = st.radio("Exercise-Induced Angina", options=[1, 0], format_func=lambda x: "Yes" if x == 1 else "No")
    oldpeak = st.slider("ST Depression Induced by Exercise", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
    slope = st.selectbox("Slope of Peak Exercise ST Segment", options=[0, 1, 2])
    ca = st.slider("Number of Major Vessels", min_value=0, max_value=4, value=0)
    thal = st.selectbox("Thalassemia Type", options=[0, 1, 2], format_func=lambda x: ["Normal", "Fixed Defect", "Reversible Defect"][x])

    # Submit button
    submit_button = st.form_submit_button("Predict")

# Make predictions when the form is submitted
if submit_button:
    # Prepare input data for the model
    user_input = np.array([age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]).reshape(1, -1)
    
    # Predict using the model
    prediction = model.predict(user_input)
    prediction_proba = model.predict_proba(user_input)

    # Display results
    st.subheader("Prediction Results")
    if prediction[0] == 1:
        st.write("The patient is likely to have heart disease.")
    else:
        st.write("The patient is unlikely to have heart disease.")
    
    st.write(f"Prediction Probability: {prediction_proba[0][1] * 100:.2f}% chance of heart disease.")
