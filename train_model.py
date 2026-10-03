# =============================================================================
# train_model.py
# Machine Learning Training Pipeline for Disease Prediction
# =============================================================================
# Workflow:
#   1. Load dataset (Training.csv)
#   2. Separate features (X) and target (y)
#   3. Encode target labels using LabelEncoder
#   4. Split data into 80% Train and 20% Test sets (stratified)
#   5. Train a Random Forest Classifier
#   6. Evaluate model on unseen test data (Accuracy, Precision, Recall, F1)
#   7. Save trained model, label encoder, and feature names to disk
# =============================================================================

import os
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

def main():
    print("=" * 70)
    print("🚀 STARTING MACHINE LEARNING TRAINING PIPELINE")
    print("=" * 70)

    # -------------------------------------------------------------------------
    # STEP 1: Load the Dataset
    # -------------------------------------------------------------------------
    data_path = os.path.join('dataset', 'Training.csv')
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at {data_path}. Please run download_dataset.py first.")

    print(f"\n[1/6] 📂 Loading dataset from '{data_path}'...")
    df = pd.read_csv(data_path)
    print(f"      Dataset loaded: {df.shape[0]} patient records, {df.shape[1]} columns.")

    # -------------------------------------------------------------------------
    # STEP 2: Separate Features (X) and Target (y)
    # -------------------------------------------------------------------------
    print("\n[2/6] ✂️  Separating Features (X) and Target (y)...")
    X = df.drop(columns=['prognosis'])
    y = df['prognosis']
    feature_names = list(X.columns)
    print(f"      Features: {len(feature_names)} symptom columns.")
    print(f"      Target:   'prognosis' with {y.nunique()} unique disease classes.")

    # -------------------------------------------------------------------------
    # STEP 3: Encode Target Disease Labels into Integers
    # -------------------------------------------------------------------------
    print("\n[3/6] 🏷️  Encoding target labels...")
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    print(f"      Successfully encoded {len(label_encoder.classes_)} unique disease classes.")

    # -------------------------------------------------------------------------
    # STEP 4: Train-Test Split (80% Training, 20% Testing)
    # -------------------------------------------------------------------------
    print("\n[4/6] 📊 Splitting dataset into Train (80%) and Test (20%) sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    print(f"      Training samples: {X_train.shape[0]}")
    print(f"      Testing samples:  {X_test.shape[0]} (unseen during training)")

    # -------------------------------------------------------------------------
    # STEP 5: Train Random Forest Classifier
    # -------------------------------------------------------------------------
    print("\n[5/6] 🌲 Training Random Forest Classifier...")
    # n_estimators=100: Build 100 decision trees
    # random_state=42: Ensure reproducible results
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    print("      Model training completed successfully!")

    # -------------------------------------------------------------------------
    # STEP 6: Model Evaluation on Unseen Test Data
    # -------------------------------------------------------------------------
    print("\n[6/6] 📈 Evaluating model performance on Test Data...")
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

    print("\n" + "=" * 70)
    print("🏆 MODEL EVALUATION METRICS (TEST SET)")
    print("=" * 70)
    print(f"  Accuracy:        {acc * 100:.2f}%  (Overall correct classifications)")
    print(f"  Precision:       {prec * 100:.2f}%  (Weighted average precision)")
    print(f"  Recall:          {rec * 100:.2f}%  (Weighted average recall)")
    print(f"  F1-Score:        {f1 * 100:.2f}%  (Harmonic mean of precision & recall)")
    print("=" * 70)

    # -------------------------------------------------------------------------
    # STEP 7: Save Trained Model and Preprocessing Objects
    # -------------------------------------------------------------------------
    os.makedirs('model', exist_ok=True)

    model_file = os.path.join('model', 'disease_model.pkl')
    encoder_file = os.path.join('model', 'label_encoder.pkl')
    features_file = os.path.join('model', 'feature_names.pkl')

    print("\n💾 Saving artifacts to 'model/' directory...")
    joblib.dump(model, model_file)
    joblib.dump(label_encoder, encoder_file)
    joblib.dump(feature_names, features_file)

    print(f"  ✅ Model saved to:          {model_file}")
    print(f"  ✅ Label Encoder saved to:  {encoder_file}")
    print(f"  ✅ Feature list saved to:   {features_file}")

    print("\n" + "=" * 70)
    print("🎉 STAGE 5 COMPLETE: MODEL READY FOR INFERENCE!")
    print("=" * 70)

if __name__ == '__main__':
    main()
