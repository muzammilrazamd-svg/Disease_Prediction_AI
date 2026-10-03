# =============================================================================
# app.py
# Disease Prediction Web Application using Streamlit
# =============================================================================

import streamlit as st
import joblib
import numpy as np
import pandas as pd
import os

# Page Configuration
st.set_page_config(
    page_title="Disease Prediction AI",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .prediction-box {
        background-color: #e8f4f8;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
    }
    .warning-box {
        background-color: #fff3cd;
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid #ffc107;
        margin-top: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# Load Model Artifacts
@st.cache_resource
def load_model_artifacts():
    """Load trained model, label encoder, and feature names"""
    try:
        model = joblib.load(os.path.join('model', 'disease_model.pkl'))
        label_encoder = joblib.load(os.path.join('model', 'label_encoder.pkl'))
        feature_names = joblib.load(os.path.join('model', 'feature_names.pkl'))
        return model, label_encoder, feature_names
    except FileNotFoundError:
        st.error("⚠️ Model files not found. Please run `train_model.py` first.")
        st.stop()

model, label_encoder, feature_names = load_model_artifacts()

# Disease Information (Educational descriptions)
DISEASE_INFO = {
    "Common Cold": "A viral infection of the upper respiratory tract. Rest, fluids, and over-the-counter medications can help.",
    "Diabetes": "A chronic condition affecting blood sugar regulation. Requires medical management and lifestyle changes.",
    "Malaria": "A parasitic infection transmitted by mosquitoes. Requires immediate medical treatment.",
    "Dengue": "A mosquito-borne viral infection common in tropical regions. Requires medical attention.",
    "Hypertension": "High blood pressure, often called the 'silent killer'. Requires monitoring and treatment.",
    "Migraine": "A neurological condition causing severe headaches. Various treatments available.",
    "Tuberculosis": "A bacterial infection primarily affecting the lungs. Requires long-term antibiotic treatment.",
    "Pneumonia": "An infection that inflames air sacs in the lungs. Requires medical treatment.",
}

# Main Title
st.markdown('<h1 class="main-header">🩺 Disease Prediction AI</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Machine Learning-Based Disease Prediction from Symptoms</p>', unsafe_allow_html=True)

# Sidebar: Patient Information
st.sidebar.header("👤 Patient Information")
patient_name = st.sidebar.text_input("Name (Optional)", placeholder="Enter patient name")
patient_age = st.sidebar.number_input("Age", min_value=1, max_value=120, value=30)
patient_gender = st.sidebar.selectbox("Gender", ["Male", "Female", "Other"])

st.sidebar.markdown("---")
st.sidebar.markdown("### ℹ️ About")
st.sidebar.info(
    "This is an **educational AI/ML project** demonstrating disease prediction "
    "using a Random Forest classification model trained on symptom data."
)

# Main Content Area
tab1, tab2 = st.tabs(["🔍 Prediction", "📊 Model Info"])

with tab1:
    st.subheader("Select Patient Symptoms")

    # Organize symptoms into columns for better UX
    st.markdown("**Check all symptoms that apply:**")

    # Create symptom selection in multiple columns
    num_cols = 4
    cols = st.columns(num_cols)

    selected_symptoms = []
    for idx, symptom in enumerate(feature_names):
        col = cols[idx % num_cols]
        # Convert underscore symptom names to readable format
        readable_symptom = symptom.replace('_', ' ').title()
        if col.checkbox(readable_symptom, key=symptom):
            selected_symptoms.append(symptom)

    st.markdown("---")

    # Predict Button
    if st.button("🔍 Predict Disease", type="primary", use_container_width=True):
        if len(selected_symptoms) == 0:
            st.warning("⚠️ Please select at least one symptom.")
        else:
            # Prepare input vector (all zeros initially)
            input_df = pd.DataFrame([np.zeros(len(feature_names), dtype=int)], columns=feature_names)

            # Set selected symptoms to 1
            for symptom in selected_symptoms:
                input_df[symptom] = 1

            # Make prediction
            with st.spinner("🔄 Analyzing symptoms..."):
                prediction_encoded = model.predict(input_df)[0]
                predicted_disease = label_encoder.inverse_transform([prediction_encoded])[0]

                # Get prediction probabilities
                probabilities = model.predict_proba(input_df)[0]
                confidence = probabilities[prediction_encoded] * 100

                # Get top 3 predictions
                top_3_indices = np.argsort(probabilities)[::-1][:3]

            # Display Results
            st.markdown("### 🎯 Prediction Results")

            col1, col2 = st.columns([2, 1])

            with col1:
                st.markdown(f'<div class="prediction-box">', unsafe_allow_html=True)
                if patient_name:
                    st.markdown(f"**Patient:** {patient_name}")
                st.markdown(f"**Selected Symptoms:** {len(selected_symptoms)} symptoms")
                st.markdown(f"**Primary Prediction:** 🩺 **{predicted_disease}**")
                st.markdown(f"**Model Confidence:** {confidence:.1f}%")

                # Disease description if available
                if predicted_disease in DISEASE_INFO:
                    st.markdown(f"**About {predicted_disease}:**")
                    st.markdown(f"*{DISEASE_INFO[predicted_disease]}*")

                st.markdown('</div>', unsafe_allow_html=True)

            with col2:
                st.markdown("**Top 3 Possibilities:**")
                for rank, idx in enumerate(top_3_indices, 1):
                    disease_name = label_encoder.inverse_transform([idx])[0]
                    prob = probabilities[idx] * 100
                    if prob > 0.1:  # Only show if probability > 0.1%
                        st.metric(
                            label=f"{rank}. {disease_name}",
                            value=f"{prob:.1f}%"
                        )

            # Display selected symptoms
            st.markdown("**Symptoms Analyzed:**")
            symptom_display = ", ".join([s.replace('_', ' ').title() for s in selected_symptoms])
            st.text(symptom_display)

with tab2:
    st.subheader("📊 Model Information")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Algorithm", "Random Forest")
        st.metric("Number of Trees", "100")

    with col2:
        st.metric("Training Samples", "3,936")
        st.metric("Test Accuracy", "99.49%")

    with col3:
        st.metric("Disease Classes", "41")
        st.metric("Symptom Features", "132")

    st.markdown("---")

    st.markdown("### 🧠 How It Works")
    st.markdown("""
    1. **Data Collection:** The model was trained on 4,920 patient records with 132 symptoms and 41 diseases.
    2. **Feature Input:** User selects symptoms experienced by the patient.
    3. **ML Prediction:** A Random Forest classifier (ensemble of 100 decision trees) analyzes symptom patterns.
    4. **Result Output:** The model predicts the most likely disease with a confidence score.
    """)

    st.markdown("### 📚 Technologies Used")
    st.markdown("""
    - **Python 3.13** — Programming language
    - **Pandas & NumPy** — Data manipulation
    - **Scikit-learn** — Machine learning (Random Forest)
    - **Streamlit** — Web application framework
    - **Joblib** — Model serialization
    """)

# Medical Disclaimer (Always visible at bottom)
st.markdown("---")
st.markdown('<div class="warning-box">', unsafe_allow_html=True)
st.markdown("### ⚠️ Medical Disclaimer")
st.markdown("""
**This application is an educational machine learning demonstration and is NOT a medical diagnostic system.**

- Predictions are based on a machine learning model trained on educational data.
- This system does NOT replace professional medical advice, diagnosis, or treatment.
- Always consult a qualified healthcare professional for any health concerns.
- Do NOT use this system to make medical decisions.
- In case of emergency, call your local emergency services immediately.
""")
st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown(
    '<p style="text-align: center; color: #999; font-size: 0.9rem;">'
    '🎓 B.Tech CSE — 3rd Year AI/ML Internship Project | '
    'Developed as an Educational ML Demonstration'
    '</p>',
    unsafe_allow_html=True
)
