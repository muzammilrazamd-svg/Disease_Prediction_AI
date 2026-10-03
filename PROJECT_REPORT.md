# 🩺 Disease Prediction from Symptoms Using Machine Learning
## Comprehensive Project Report & Internship Documentation

**Author:** 3rd Year B.Tech CSE Student  
**Project Category:** Artificial Intelligence & Machine Learning / Healthcare Informatics  
**Academic Year:** 2026  

---

## 1. Title
**Disease Prediction from Symptoms Using Machine Learning: An Educational Supervised Classification Approach**

---

## 2. Abstract
Early identification of potential health conditions plays a vital role in medical awareness and education. This project implements a machine learning-based educational application that takes patient-reported symptoms and predicts possible disease categories using a multi-class Random Forest classification model. Trained on 4,920 synthetic patient records spanning 132 distinct symptoms across 41 disease categories, the model achieves a test classification accuracy of **99.49%**, with weighted precision of **99.52%** and recall of **99.49%**. The system is packaged into an interactive Streamlit web application providing top-3 candidate diseases along with confidence scores and clear educational disclaimers.

---

## 3. Introduction
Healthcare systems worldwide face challenges in triage and early symptom assessment. While machine learning cannot replace qualified medical professionals, intelligent systems can serve as educational tools and decision-support demonstrations. This project explores how supervised classification algorithms can map complex combinations of binary symptom indicators into categorical disease classifications.

---

## 4. Problem Statement
Given an arbitrary set of $N$ binary symptom indicators $S = \{s_1, s_2, \dots, s_{132}\} \in \{0, 1\}^{132}$ reported by a patient, classify the symptom profile into one of 41 candidate disease categories $D \in \{d_1, d_2, \dots, d_{41}\}$ and output the associated prediction probabilities while maintaining clear non-diagnostic boundaries.

---

## 5. Motivation
Many patients struggle to articulate symptom combinations or understand how multiple overlapping symptoms relate to different medical conditions. Developing a transparent, explainable machine learning pipeline with an intuitive web interface demonstrates the practical application of classification algorithms in health informatics.

---

## 6. Objectives
1. Build a complete end-to-end Machine Learning pipeline (Data ingestion $\rightarrow$ Preprocessing $\rightarrow$ Training $\rightarrow$ Evaluation $\rightarrow$ Inference).
2. Train a supervised classification model on a structured multi-disease symptom dataset.
3. Evaluate model performance using Accuracy, Precision, Recall, and F1-Score metrics.
4. Serialize and persist model artifacts using Joblib for low-latency web deployment.
5. Develop an intuitive, responsive Streamlit web application featuring confidence scores and prominent medical disclaimers.

---

## 7. Existing System vs. Proposed System

| Feature | Existing Systems | Proposed System |
|---|---|---|
| Approach | Rule-based if/else logic or search keywords | Machine Learning (Random Forest Classifier) |
| Scalability | Difficult to scale beyond a few rules | Scales automatically across 132 features & 41 diseases |
| Probabilistic Output | Deterministic binary match | Multi-class probability distribution & Top-3 suggestions |
| Interface | Complex or static forms | Modern, responsive Streamlit web UI |
| Latency | Manual table lookups | Sub-10ms model inference |

---

## 8. Technologies Used
- **Programming Language:** Python 3.13
- **Data Manipulation:** Pandas (v3.0.6), NumPy (v2.5.3)
- **Machine Learning:** Scikit-Learn (v1.9.1)
- **Model Serialization:** Joblib (v1.6.0)
- **Web Framework:** Streamlit (v1.64.0)
- **IDE:** Visual Studio Code (Windows 11)

---

## 9. Dataset Description
- **Origin:** Educational symptom-disease dataset based on Kaggle's open-access repository.
- **Records:** 4,920 patient instances.
- **Features ($X$):** 132 binary symptom columns (0 = absent, 1 = present).
- **Target ($y$):** 41 categorical disease classes (e.g., Malaria, Dengue, Diabetes, Hypertension).
- **Class Balance:** Balanced distribution with 120 records per disease class.
- **Missing Values:** Zero missing/null values across all columns.

---

## 10. Methodology & ML Pipeline

```
[Raw CSV Dataset]
       │
       ▼
[Feature/Target Separation] ──► X (132 features), y (1 target)
       │
       ▼
[Label Encoding] ──────────────► Categorical text → Integers [0..40]
       │
       ▼
[Stratified Train-Test Split] ─► 80% Training (3936), 20% Testing (984)
       │
       ▼
[Random Forest Training] ──────► 100 Decision Trees with Bootstrap Aggregation
       │
       ▼
[Model Evaluation] ────────────► Accuracy: 99.49%, F1: 99.49%
       │
       ▼
[Model Serialization] ─────────► Saved as .pkl files via Joblib
       │
       ▼
[Streamlit Web App] ───────────► Real-time inference & UI rendering
```

---

## 11. Machine Learning Algorithm: Random Forest Classifier

### Why Random Forest?
1. **Ensemble Learning (Bagging):** Builds 100 independent decision trees using bootstrap samples of the training data.
2. **Feature Subsampling:** Each tree considers a random subset of features at each split, reducing correlation between individual trees.
3. **Variance Reduction:** Majority voting over 100 trees minimizes the risk of overfitting compared to a single deep decision tree.
4. **Probability Estimation:** Computes class probabilities based on the proportion of trees voting for each class.

---

## 12. Experimental Results & Evaluation

Evaluation conducted on **984 unseen test records** (20% held-out test split):

- **Overall Accuracy:** 99.49%
- **Weighted Precision:** 99.52%
- **Weighted Recall:** 99.49%
- **Weighted F1-Score:** 99.49%

---

## 13. System Architecture & Component Design

```
+-------------------------------------------------------------+
|                      USER BROWSER                           |
|       (Streamlit UI: Patient Info, Symptom Checkboxes)      |
+------------------------------+------------------------------+
                               │ HTTP Request / User Actions
                               ▼
+-------------------------------------------------------------+
|                     STREAMLIT BACKEND (app.py)              |
|  - Input Validation & Vectorization (132-dim binary vector) |
|  - Joblib Artifact Loading (Cached via @st.cache_resource)  |
+------------------------------+------------------------------+
                               │ Vector Input
                               ▼
+-------------------------------------------------------------+
|                SERIALIZED ML ENGINE (model/)                |
|  - disease_model.pkl (RandomForestClassifier, 100 trees)    |
|  - label_encoder.pkl (LabelEncoder)                         |
|  - feature_names.pkl (Feature List)                         |
+-------------------------------------------------------------+
```

---

## 14. Ethical, Privacy & Medical Considerations
1. **Educational Purpose:** The software is designed strictly for academic demonstration and not for clinical diagnosis.
2. **Privacy Protection:** Patient names and demographic details are used solely for UI greeting and are never stored or logged to any database.
3. **Non-Diagnostic Disclaimer:** Prominently featured on every screen to ensure users consult certified physicians for actual medical symptoms.

---

## 15. Limitations & Future Scope
- **Limitations:** Binary symptom representation does not capture symptom severity or progression timeline. The dataset is educational and synthetic.
- **Future Scope:** Integration of continuous physiological metrics (e.g., blood pressure, lab values), natural language processing for free-form symptom entry, and multi-language support.

---

## 16. Conclusion
The Disease Prediction AI project successfully demonstrates the end-to-end design, implementation, and deployment of a supervised machine learning classification system. With a Random Forest ensemble model achieving over 99% accuracy on the test partition and a responsive Streamlit web application, the project meets all 3rd-year engineering internship criteria.
