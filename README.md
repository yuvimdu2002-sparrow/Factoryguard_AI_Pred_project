# 🏭 FactoryGuard AI – Machine Failure Prediction

FactoryGuard AI is a **machine learning–based predictive maintenance project** that predicts whether an industrial robotic arm will fail within the **next 24 hours** using sensor data.  
The project focuses on **imbalanced data handling**, **high-precision prediction**, and **model explainability**.

---

## 🎯 Project Objective

- Predict machine failure within the next 24 hours
- Handle highly imbalanced industrial data
- Reduce false alarms by maintaining high precision
- Explain model predictions using SHAP
- Build an end-to-end ML pipeline

---

## 📊 Dataset Overview

**Sensor Features:**
- Temperature (°C)
- Vibration (RMS mm/s)
- Pressure (bar)
- RPM
- Load percentage
- Error count
- Maintenance history
- Humidity

**Target Variable:**
- `failure_in_next_24h`
  - `1` → Failure expected within 24 hours
  - `0` → No failure

## 🧱 Project Structure
FactoryGuard-AI_Pred_project/
├── data/
│ ├── raw/
│ │ └── factoryguard_synthetic_500.csv
│ └── processed/
│ └── features_engineering_output.csv
├── model/
│ ├── baseline_logistic_gridsearch.joblib
│ ├── features_engineering.joblib
│ └── xgboost_tuned.joblib
├── notebook/
│ └── EDA.ipynb
├── report/
│ ├── feature_selection_report.csv
│ ├── shap_global_feature_importance.png
│ └── shap_local_failure_explanation.png
├── src/
│ ├── init.py
│ ├── feature_engineering.py
│ ├── feature_selection.py
│ ├── model_evaluation.py
│ ├── Shap.py
│ ├── train_baseline_logreg.py
│ └── train_xgboost_optuna.py
├── .gitattributes
├── requirements.txt
└── README.md

---

## 🔧 Feature Engineering

- Rolling Mean (1h, 6h, 12h)
- Rolling Standard Deviation
- Exponential Moving Average (EMA)
- Lag features (t-1, t-2)

Target label:


Unnecessary features were removed using feature selection techniques.

---

## 🎯 Feature Selection Methods

- Filter Method (ANOVA F-test)
- Wrapper Method (RFE – Recursive Feature Elimination)
- Embedded Method (Random Forest Feature Importance)

---

## 🤖 Models Used

### 1️⃣ Logistic Regression (Baseline)
- `class_weight = "balanced"`
- Hyperparameter tuning using GridSearchCV
- Evaluation metric: **PR-AUC**

### 2️⃣ XGBoost (Final Model)
- Hyperparameter tuning using Optuna
- Handles class imbalance using `scale_pos_weight`
- Optimized for high precision

---

## 📈 Model Evaluation

- Primary Metric: **PR-AUC (Precision–Recall AUC)**
- Test-set evaluation only
- Custom threshold selected for ≥90% precision
- Classification report generated

---

## 🔍 Model Explainability (SHAP)

- **Global Explanation:** Important features affecting failures
- **Local Explanation:** Reason behind individual failure predictions

Generated outputs:
- `shap_global_feature_importance.png`
- `shap_local_failure_explanation.png`

---

## 🛠️ Installation

```bash
pip install -r requirements.txt


## 🧱 Project Structure

