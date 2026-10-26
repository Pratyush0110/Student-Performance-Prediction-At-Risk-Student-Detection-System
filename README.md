# Student Performance Prediction & At-Risk Student Detection System

🚀 **[🌐 LIVE DEMO — Try the App](https://student-risk-pratyush.streamlit.app/)**

An ML-powered application that predicts student academic risk and helps identify students who may need additional support.

## Features
- Student risk prediction
- Machine learning model comparison
- Model performance evaluation
- Interactive Streamlit dashboard

## Tech Stack
Python · Pandas · Scikit-learn · Streamlit

## Run Locally
```bash
pip install -r requirements.txt
streamlit run app.py
```# Student Performance Prediction & At-Risk Student Detection System

A machine-learning web application that estimates whether a student may be at risk of academic underperformance.

## Features
- Interactive Streamlit dashboard
- Student risk prediction
- Comparison of Logistic Regression, Decision Tree, KNN, and Random Forest
- Model comparison metrics, confusion matrix, and classification report

## Technology
Python, Pandas, NumPy, Scikit-learn, Matplotlib, Joblib, Streamlit

## Run locally
Install dependencies:

```bash
pip install -r requirements.txt
```

Launch the app:

```bash
streamlit run app.py
```

## Dataset
UCI Student Performance dataset:
https://archive.ics.uci.edu/dataset/320/student+performance

The target is defined as at-risk when G3 is below 10. G1, G2, and G3 are excluded from model input features to avoid direct grade leakage.

## Evaluation
The selected Logistic Regression model achieved 70.9% accuracy, 65.4% recall for the at-risk class, and an at-risk F1 score of 59.6% on the held-out test set.

Results may differ on other student populations. This is an educational prototype, not a definitive assessment tool.

## Author
Pratyush Singh Yadav
