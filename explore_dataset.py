# =============================================================
# explore_dataset.py
# Script to explore and understand our dataset
# Run this to see what our data looks like before building the model
# =============================================================

import pandas as pd

# Load the dataset
df = pd.read_csv('dataset/Training.csv')

# ─── 1. Basic Shape ───
print("=" * 60)
print("📊 DATASET OVERVIEW")
print("=" * 60)
print(f"Number of rows (patient cases):    {df.shape[0]}")
print(f"Number of columns:                 {df.shape[1]}")
print(f"Number of symptom features:        {df.shape[1] - 1}")
print(f"Target column:                     'prognosis'")

# ─── 2. Column Names ───
print("\n" + "=" * 60)
print("📋 ALL COLUMNS")
print("=" * 60)
symptom_columns = [col for col in df.columns if col != 'prognosis']
print(f"\nSymptom columns ({len(symptom_columns)} total):")
for i, col in enumerate(symptom_columns, 1):
    print(f"  {i:3d}. {col}")
print(f"\nTarget column: prognosis")

# ─── 3. Sample Rows ───
print("\n" + "=" * 60)
print("👀 FIRST 5 ROWS (showing symptoms that are = 1)")
print("=" * 60)
for idx in range(5):
    row = df.iloc[idx]
    disease = row['prognosis']
    active_symptoms = [col for col in symptom_columns if row[col] == 1]
    print(f"\nRow {idx + 1}: Disease = '{disease}'")
    print(f"  Active symptoms: {', '.join(active_symptoms)}")

# ─── 4. Data Types ───
print("\n" + "=" * 60)
print("🔢 DATA TYPES")
print("=" * 60)
print(f"Symptom columns: all {df[symptom_columns].dtypes.unique()[0]} (0 or 1)")
print(f"Target column: {df['prognosis'].dtype} (disease name text)")

# ─── 5. Missing Values ───
print("\n" + "=" * 60)
print("❓ MISSING VALUES")
print("=" * 60)
missing = df.isnull().sum().sum()
print(f"Total missing values: {missing}")
if missing == 0:
    print("Great! No missing values — no imputation needed.")

# ─── 6. Target Classes ───
print("\n" + "=" * 60)
print("🏥 DISEASE CLASSES (Target)")
print("=" * 60)
disease_counts = df['prognosis'].value_counts()
print(f"Number of unique diseases: {disease_counts.shape[0]}")
print(f"\nCases per disease:")
for disease, count in disease_counts.items():
    print(f"  {disease:45s} → {count} cases")

# ─── 7. Feature Value Range ───
print("\n" + "=" * 60)
print("📈 SYMPTOM VALUE RANGE")
print("=" * 60)
print(f"Minimum value: {df[symptom_columns].min().min()}")
print(f"Maximum value: {df[symptom_columns].max().max()}")
print(f"Unique values: {sorted(df[symptom_columns].stack().unique())}")
print("(Binary: 0 = symptom absent, 1 = symptom present)")

# ─── 8. Most Common Symptoms ───
print("\n" + "=" * 60)
print("🔥 TOP 15 MOST COMMON SYMPTOMS (across all cases)")
print("=" * 60)
symptom_freq = df[symptom_columns].sum().sort_values(ascending=False)
for symptom, count in symptom_freq.head(15).items():
    pct = count / len(df) * 100
    print(f"  {symptom:35s} → {count:5d} cases ({pct:.1f}%)")

print("\n" + "=" * 60)
print("✅ Dataset exploration complete!")
print("=" * 60)
