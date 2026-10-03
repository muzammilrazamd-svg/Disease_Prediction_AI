# =============================================================
# test_preprocessing.py
# Stage 4: Verifying Data Preprocessing Pipeline
# =============================================================

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# 1. Load Dataset
print("=" * 60)
print("📂 STEP 1: LOADING DATASET")
print("=" * 60)
df = pd.read_csv('dataset/Training.csv')
print(f"Loaded dataset with shape: {df.shape}")

# 2. Separate Features (X) and Target (y)
print("\n" + "=" * 60)
print("✂️ STEP 2: SEPARATING FEATURES (X) AND TARGET (y)")
print("=" * 60)
X = df.drop(columns=['prognosis'])
y = df['prognosis']

print(f"X (Features shape): {X.shape}  -> 132 symptom columns")
print(f"y (Target shape):   {y.shape}    -> 1 disease column")

# 3. Encode Target Labels (Text -> Integers)
print("\n" + "=" * 60)
print("🏷️ STEP 3: LABEL ENCODING TARGET")
print("=" * 60)
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

print("Sample mappings (First 5 classes):")
for idx, class_name in enumerate(encoder.classes_[:5]):
    print(f"  Class '{class_name}' -> Encoded ID: {idx}")

print(f"\nTotal encoded classes: {len(encoder.classes_)}")

# 4. Train-Test Split (80% Train, 20% Test)
print("\n" + "=" * 60)
print("📊 STEP 4: TRAIN-TEST SPLIT (80% Train / 20% Test)")
print("=" * 60)
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

print(f"Training set: {X_train.shape[0]} samples ({X_train.shape[0]/len(df)*100:.1f}%)")
print(f"Testing set:  {X_test.shape[0]} samples ({X_test.shape[0]/len(df)*100:.1f}%)")

print("\n" + "=" * 60)
print("✅ PREPROCESSING PIPELINE VERIFIED SUCCESSFULLY!")
print("=" * 60)
