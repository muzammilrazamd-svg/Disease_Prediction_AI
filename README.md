# 🩺 Disease Prediction from Symptoms using Machine Learning

An educational machine learning web application that predicts possible diseases
based on patient symptoms using a trained classification model.

> ⚠️ **Medical Disclaimer:** This application is an educational machine-learning
> demonstration and is NOT a medical diagnostic system. Predictions must not be
> used for medical decisions. Always consult a qualified healthcare professional.

## Technologies Used

- Python 3.13
- Pandas & NumPy (data processing)
- Scikit-learn (machine learning)
- Streamlit (web interface)
- Joblib (model serialization)

## How to Run

1. Create and activate virtual environment:
   ```
   python -m venv venv
   venv\Scripts\activate
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Train the model:
   ```
   python train_model.py
   ```

4. Run the application:
   ```
   streamlit run app.py
   ```

## Project Structure

```
Disease_Prediction_AI/
├── dataset/          # Training dataset
├── model/            # Saved trained model
├── train_model.py    # ML training script
├── app.py            # Streamlit web application
├── requirements.txt  # Python dependencies
├── README.md         # Project documentation
└── .gitignore        # Git ignore rules
```

## Author

B.Tech CSE — 3rd Year Internship Project
