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

```
---

## Web Application

The Flask web app allows users to:

- Enter sensor values
- Upload JSON input
- Click Predict
- View probability result

---

## Technologies Used

- Python  
- Flask  
- Scikit-Learn  
- XGBoost  
- Pandas  
- NumPy  
- SHAP  
- HTML  
- CSS  
- Joblib  

---

## Project Structure

```text
FactoryGuard_AI_Pred_Project/
│
├── data/
│   ├── processed/
│   │      └── factoryguard_synthetic_500.csv
│   └── raw/
│          └── features_engineering_output.csv
│
├── json_input/
│   └── input 1.json
│   └── input 2.json
│
│
├── model/
│   ├── baseline_logistic_gridsearch.joblib
│   ├── feature_columns.joblib
│   ├── features_engineering.joblib
│   └── xgboost_optuna_tuned.joblib
│
├── notebook/
│   └── EDA.ipynb
│
├── report/
│   ├── feature_selection_report.csv
│   ├── model_evaluation_comparison.csv
│   ├── pr_curve_comparison.png
│   ├── shap_global_feature_importance.jpeg
│   └── shap_local_failure_explanation.png
│
├── src/
│   ├── feature_engineering.py
│   ├── feature_selection.py
│   ├── model_evaluation.py
│   ├── shap.py
│   ├── train_baseline_logreg.py
│   └── train_xgboost_optuna.py
│
├── static/
│   ├── factoryguard_ai_image.png
│   └── style_text.css
│
├── templates/
│   └── pred.html
│
├── app.py
├── requirements.txt
└── README.md
```

---

## How to Run the Project

1. Clone the repository  
2. Install dependencies  
   pip install -r requirements.txt  
3. Run the application  
   python app.py  
4. Open browser and visit  
   http://127.0.0.1:5000  

---

## Project Features

- Machine failure prediction  
- JSON input support  
- Simple web UI  
- SHAP explainability graphs  
- Clean modular folder structure  
- High precision ML model  

---

## Future Improvements

- Live sensor data integration  
- Cloud deployment  
- Dashboard analyics
- Email and SMS alerts
  
---

---

## Team / Contributors

This project was developed as a **team project** as part of learning and portfolio development.

Team Members:

- Yuvaraj A
- Somashekara T.R
- Dhanunjay Kadapa
- Satyajit Maharana

Each member contributed to different parts such as data preprocessing, model training, web development, and documentation.

---

## License

This project is developed solely for *educational and portfolio purposes* as part of an internship/project work. No commercial use is intended. Feel free to explore and learn from the code.
