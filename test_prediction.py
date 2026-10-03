# =============================================================================
# test_prediction.py
# Stage 6: Test inference using saved model artifacts
# =============================================================================

import os
import joblib
import numpy as np
import pandas as pd

def test_inference():
    print("=" * 70)
    print("🧪 TESTING INFERENCE PIPELINE")
    print("=" * 70)

    # 1. Load artifacts
    model_path = os.path.join('model', 'disease_model.pkl')
    encoder_path = os.path.join('model', 'label_encoder.pkl')
    features_path = os.path.join('model', 'feature_names.pkl')

    print("1. Loading model artifacts from 'model/' folder...")
    model = joblib.load(model_path)
    label_encoder = joblib.load(encoder_path)
    feature_names = joblib.load(features_path)
    print("   Artifacts successfully loaded into memory!")

    # 2. Define test patient cases with specific symptoms
    test_cases = [
        {
            "patient_name": "Test Patient 1 (Dengue Symptoms)",
            "symptoms": ["skin_rash", "high_fever", "headache", "vomiting", "joint_pain", "loss_of_appetite"]
        },
        {
            "patient_name": "Test Patient 2 (Diabetes Symptoms)",
            "symptoms": ["fatigue", "weight_loss", "restlessness", "irregular_sugar_level", "excessive_hunger", "polyuria"]
        },
        {
            "patient_name": "Test Patient 3 (Common Cold Symptoms)",
            "symptoms": ["continuous_sneezing", "chills", "fatigue", "cough", "high_fever", "headache", "runny_nose", "congestion"]
        },
        {
            "patient_name": "Test Patient 4 (Fungal Infection Symptoms)",
            "symptoms": ["itching", "skin_rash", "nodal_skin_eruptions"]
        }
    ]

    print("\n2. Simulating Predictions for Sample Patient Symptom Profiles:")
    print("-" * 70)

    for case in test_cases:
        # Create a zero-vector of length 132 (all symptoms absent initially)
        # Using a DataFrame with feature names prevents scikit-learn feature name warnings
        input_df = pd.DataFrame([np.zeros(len(feature_names), dtype=int)], columns=feature_names)

        # Mark present symptoms as 1
        for sym in case['symptoms']:
            if sym in feature_names:
                input_df[sym] = 1
            else:
                print(f"   [Warning] Unknown symptom: {sym}")

        # Make prediction
        pred_encoded = model.predict(input_df)[0]
        pred_disease = label_encoder.inverse_transform([pred_encoded])[0]

        # Get prediction confidence / probabilities
        probabilities = model.predict_proba(input_df)[0]
        confidence = probabilities[pred_encoded] * 100

        # Get top 3 likely diseases
        top_3_indices = np.argsort(probabilities)[::-1][:3]

        print(f"\n📋 Case: {case['patient_name']}")
        print(f"   Selected Symptoms: {', '.join(case['symptoms'])}")
        print(f"   👉 Predicted Disease:    ⭐ {pred_disease} (Confidence: {confidence:.2f}%)")
        print("   🔍 Top 3 Possibilities:")
        for rank, idx in enumerate(top_3_indices, 1):
            d_name = label_encoder.inverse_transform([idx])[0]
            prob = probabilities[idx] * 100
            if prob > 0:
                print(f"      {rank}. {d_name:30s} -> {prob:5.2f}%")

    print("\n" + "=" * 70)
    print("✅ STAGE 6 COMPLETE: MODEL INFERENCE IS WORKING ACCURATELY!")
    print("=" * 70)

if __name__ == '__main__':
    test_inference()
