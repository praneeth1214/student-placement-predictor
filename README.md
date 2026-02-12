# 🧠 AI Career Intelligence Dashboard

Enterprise-grade AI system for evaluating **career readiness** and predicting **placement probability** using calibrated machine learning.

---

## 📌 Overview

The AI Career Intelligence Dashboard is a decision-support platform that analyzes student academic and skill-based inputs to provide:

- Career Readiness Index
- Role-specific skill alignment
- Placement probability estimation
- Stability classification for placement preparedness

The system combines interpretable machine learning with structured readiness scoring to convert raw student data into actionable career insights.

---

## 🌐 Live Demo

👉 https://student-placement-predictor-lexgnrynd3wkb7ecvwpcgy.streamlit.app/

---

## 🎯 Problem Statement

Engineering students often rely solely on CGPA to estimate placement chances.

However, real placement outcomes depend on multiple factors including:

- Technical skill alignment
- Internship exposure
- Project experience
- Academic performance

There is a need for a structured system that integrates these factors into an interpretable and data-driven career intelligence framework.

---

## 🚀 Features

- 🎯 Target Role Selection (Web, Data, ML, AI, Software)
- 🧠 Skill Assessment Matrix
- 📊 Career Readiness Index
- 🔮 ML-Based Placement Probability Prediction
- 📈 Probability Calibration for Stable Estimation
- 💼 Enterprise Dashboard UI (Streamlit + Plotly)

---

## 🏗 System Architecture

### Input Features
- CGPA
- Projects Completed
- Internships
- Skill Alignment Score
- Attendance (maintained for model compatibility)
- Backlogs (maintained for model compatibility)

### Output
- Placement Probability (0–100%)
- Stability Classification
- Career Readiness Index

---

## 🧪 Machine Learning Model

- Algorithm: Logistic Regression (L2 Regularization)
- Validation: 5-Fold Cross Validation
- Evaluation Metric: ROC-AUC
- Probability Calibration: Sigmoid Method
- Comparison Baseline: Random Forest (for evaluation only)

### Why Logistic Regression?

Interpretability is critical in educational decision systems.  
Logistic Regression provides stable, explainable probability estimates rather than black-box outputs.

---

## 🛠 Tech Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- Streamlit
- Plotly
- Joblib

---

## 📊 Model Validation

- Train/Test Split: 80/20
- 5-Fold Cross Validation
- ROC-AUC performance evaluation
- Calibrated probability outputs

---

## 🚀 Future Scope

- Integration with institutional placement databases
- Faculty-level analytics dashboard
- Real-time industry skill benchmarking
- Continuous model retraining using placement outcomes

---

## ⚠️ Limitations

- Dataset size is limited
- Predictions are probabilistic, not deterministic
- Real-world deployment requires larger institutional data

---

## 🏆 Project Context

Developed for technical expo presentation —  
designed as a scalable AI-driven career intelligence platform.
