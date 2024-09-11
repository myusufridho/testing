import streamlit as st
import joblib
import numpy as np

# Load models
random_forest_model = joblib.load('models/random_forest_model.pkl')
gbm_model = joblib.load('models/gbm_model.pkl')
svm_model = joblib.load('models/svm_model.pkl')

# Streamlit UI
st.title('Machine Learning Model Inference')

# Input form
feature1 = st.number_input('Feature 1', format="%.2f")
feature2 = st.number_input('Feature 2', format="%.2f")
# Tambahkan input untuk semua fitur yang diperlukan di sini

if st.button('Predict'):
    features = np.array([feature1, feature2])  # Tambahkan semua fitur di sini

    # Predict with Random Forest
    rf_pred = random_forest_model.predict([features])[0]

    # Predict with GBM
    gbm_pred = gbm_model.predict([features])[0]

    # Predict with SVM
    svm_pred = svm_model.predict([features])[0]

    # Display results
    st.write(f'Random Forest Prediction: {rf_pred}')
    st.write(f'GBM Prediction: {gbm_pred}')
    st.write(f'SVM Prediction: {svm_pred}')
