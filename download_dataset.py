# =============================================================
# download_dataset.py
# Helper script to create the disease-symptom dataset
# Based on the popular Kaggle "Disease Prediction Using ML" dataset
# Source: https://www.kaggle.com/datasets/kaushil268/disease-prediction-using-machine-learning
# =============================================================
# This dataset is for EDUCATIONAL purposes only.
# It does NOT represent real clinical data.
# =============================================================

import pandas as pd
import numpy as np
import os

# All 132 symptoms from the dataset
SYMPTOMS = [
    'itching', 'skin_rash', 'nodal_skin_eruptions', 'continuous_sneezing',
    'shivering', 'chills', 'joint_pain', 'stomach_pain', 'acidity',
    'ulcers_on_tongue', 'muscle_wasting', 'vomiting', 'burning_micturition',
    'spotting_urination', 'fatigue', 'weight_gain', 'anxiety',
    'cold_hands_and_feets', 'mood_swings', 'weight_loss', 'restlessness',
    'lethargy', 'patches_in_throat', 'irregular_sugar_level', 'cough',
    'high_fever', 'sunken_eyes', 'breathlessness', 'sweating', 'dehydration',
    'indigestion', 'headache', 'yellowish_skin', 'dark_urine', 'nausea',
    'loss_of_appetite', 'pain_behind_the_eyes', 'back_pain', 'constipation',
    'abdominal_pain', 'diarrhoea', 'mild_fever', 'yellow_urine',
    'yellowing_of_eyes', 'acute_liver_failure', 'fluid_overload',
    'swelling_of_stomach', 'swelled_lymph_nodes', 'malaise',
    'blurred_and_distorted_vision', 'phlegm', 'throat_irritation',
    'redness_of_eyes', 'sinus_pressure', 'runny_nose', 'congestion',
    'chest_pain', 'weakness_in_limbs', 'fast_heart_rate',
    'pain_during_bowel_movements', 'pain_in_anal_region', 'bloody_stool',
    'irritation_in_anus', 'neck_pain', 'dizziness', 'cramps', 'bruising',
    'obesity', 'swollen_legs', 'swollen_blood_vessels',
    'puffy_face_and_eyes', 'enlarged_thyroid', 'brittle_nails',
    'swollen_extremeties', 'excessive_hunger', 'extra_marital_contacts',
    'drying_and_tingling_lips', 'slurred_speech', 'knee_pain',
    'hip_joint_pain', 'muscle_weakness', 'stiff_neck', 'swelling_joints',
    'movement_stiffness', 'spinning_movements', 'loss_of_balance',
    'unsteadiness', 'weakness_of_one_body_side', 'loss_of_smell',
    'bladder_discomfort', 'foul_smell_of_urine', 'continuous_feel_of_urine',
    'passage_of_gases', 'internal_itching', 'toxic_look_(typhos)',
    'depression', 'irritability', 'muscle_pain', 'altered_sensorium',
    'red_spots_over_body', 'belly_pain', 'abnormal_menstruation',
    'dischromic_patches', 'watering_from_eyes', 'increased_appetite',
    'polyuria', 'family_history', 'mucoid_sputum', 'rusty_sputum',
    'lack_of_concentration', 'visual_disturbances',
    'receiving_blood_transfusion', 'receiving_unsterile_injections', 'coma',
    'stomach_bleeding', 'distention_of_abdomen',
    'history_of_alcohol_consumption', 'fluid_overload.1', 'blood_in_sputum',
    'prominent_veins_on_calf', 'palpitations', 'painful_walking',
    'pus_filled_pimples', 'blackheads', 'scurring', 'skin_peeling',
    'silver_like_dusting', 'small_dents_in_nails', 'inflammatory_nails',
    'blister', 'red_sore_around_nose', 'yellow_crust_ooze'
]

# All 41 diseases and their associated symptom indices (0-based)
# Each disease has a specific set of symptoms that characterize it
DISEASE_SYMPTOMS = {
    'Fungal infection': [0, 1, 2],
    'Allergy': [3, 4, 5, 24, 102],
    'GERD': [8, 11, 31, 34, 24, 38],
    'Chronic cholestasis': [0, 11, 34, 35, 32, 33, 43, 22, 14],
    'Drug Reaction': [0, 1, 14, 31, 24, 25, 11, 98],
    'Peptic ulcer diseae': [11, 30, 35, 40, 7, 39],
    'AIDS': [96, 25, 19, 24, 14, 41, 1, 10, 47],
    'Diabetes': [14, 19, 21, 23, 74, 50, 103, 49, 27, 68],
    'Gastroenteritis': [11, 27, 29, 31, 14, 40],
    'Bronchial Asthma': [14, 24, 25, 27, 50, 104],
    'Hypertension': [31, 57, 65, 64, 17],
    'Migraine': [8, 30, 31, 35, 65, 49, 94, 36],
    'Cervical spondylosis': [37, 64, 65, 57, 83],
    'Paralysis (brain hemorrhage)': [11, 31, 87, 86, 85],
    'Jaundice': [0, 11, 14, 25, 32, 33, 43],
    'Malaria': [4, 11, 25, 31, 96, 34, 29, 14, 10],
    'Chicken pox': [0, 1, 14, 21, 25, 31, 41, 47, 98, 97],
    'Dengue': [1, 11, 14, 25, 31, 65, 34, 36, 96, 7],
    'Typhoid': [4, 11, 14, 25, 31, 34, 39, 40, 93],
    'hepatitis A': [11, 14, 34, 35, 32, 33, 43, 25, 7, 41],
    'Hepatitis B': [0, 14, 21, 34, 35, 32, 33, 43, 25, 109, 110],
    'Hepatitis C': [14, 35, 34, 32, 33, 43, 11],
    'Hepatitis D': [11, 14, 34, 35, 32, 33, 43, 25, 65],
    'Hepatitis E': [11, 14, 34, 35, 32, 33, 43, 25, 7, 39],
    'Alcoholic hepatitis': [11, 14, 34, 35, 32, 33, 43, 46, 113],
    'Tuberculosis': [4, 11, 14, 24, 25, 19, 28, 41, 50, 15, 112],
    'Common Cold': [3, 4, 5, 14, 24, 25, 31, 51, 45, 54, 55, 96],
    'Pneumonia': [4, 24, 25, 27, 14, 50, 56, 28, 105],
    'Dimorphic hemmorhoids(piles)': [38, 59, 40, 60, 61, 62],
    'Heart attack': [11, 27, 28, 56, 57],
    'Varicose veins': [14, 66, 67, 68, 69, 115, 116],
    'Hypothyroidism': [14, 15, 18, 70, 17, 71, 72, 73, 65, 95, 16],
    'Hyperthyroidism': [14, 18, 19, 28, 57, 65, 40, 102, 95],
    'Hypoglycemia': [11, 14, 16, 28, 31, 65, 77, 95, 57, 49],
    'Osteoarthristis': [78, 79, 82, 83, 80, 84],
    'Arthritis': [10, 6, 82, 83, 80, 47, 84],
    '(vertigo) Paroxymal Positional Vertigo': [11, 31, 85, 86, 84, 64],
    'Acne': [1, 119, 120, 121],
    'Urinary tract infection': [12, 14, 89, 90, 91],
    'Psoriasis': [1, 6, 122, 123, 124, 125],
    'Impetigo': [1, 25, 126, 127, 128],
}

def generate_dataset():
    """Generate the training dataset"""
    np.random.seed(42)
    rows = []

    for disease, symptom_indices in DISEASE_SYMPTOMS.items():
        # Generate 120 cases per disease (41 diseases × 120 = 4920 rows)
        for _ in range(120):
            row = [0] * len(SYMPTOMS)

            # Set the primary symptoms for this disease (always present)
            for idx in symptom_indices:
                row[idx] = 1

            # Randomly drop 0-2 non-essential symptoms (to add realistic variation)
            if len(symptom_indices) > 3:
                num_to_drop = np.random.randint(0, min(3, len(symptom_indices) - 2))
                if num_to_drop > 0:
                    drop_indices = np.random.choice(
                        symptom_indices[2:],  # never drop the first 2 key symptoms
                        size=num_to_drop,
                        replace=False
                    )
                    for idx in drop_indices:
                        row[idx] = 0

            # Occasionally add 0-1 random noise symptoms (very rarely)
            if np.random.random() < 0.1:
                noise_idx = np.random.randint(0, len(SYMPTOMS))
                if noise_idx not in symptom_indices:
                    row[noise_idx] = 1

            row.append(disease)
            rows.append(row)

    # Create DataFrame
    columns = SYMPTOMS + ['prognosis']
    df = pd.DataFrame(rows, columns=columns)

    # Shuffle the dataset
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    return df

if __name__ == '__main__':
    print("Generating disease prediction dataset...")
    print("Based on: https://www.kaggle.com/datasets/kaushil268/disease-prediction-using-machine-learning")
    print()

    df = generate_dataset()

    # Save to CSV
    output_path = os.path.join('dataset', 'Training.csv')
    df.to_csv(output_path, index=False)

    print(f"Dataset saved to: {output_path}")
    print(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"Number of symptoms (features): {df.shape[1] - 1}")
    print(f"Number of diseases (classes): {df['prognosis'].nunique()}")
    print(f"\nDiseases: {sorted(df['prognosis'].unique())}")
    print(f"\nFirst 5 rows (showing first 6 symptom columns + target):")
    display_cols = SYMPTOMS[:6] + ['prognosis']
    print(df[display_cols].head().to_string(index=False))
    print("\nDataset generated successfully!")
