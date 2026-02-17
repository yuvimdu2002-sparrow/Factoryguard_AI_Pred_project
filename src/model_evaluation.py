from sklearn.linear_model import LogisticRegression
import xgboost
import matplotlib.pyplot as plt
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    precision_recall_curve
)

# Load processed dataset

df = pd.read_csv("/content/drive/MyDrive/Zaalima project/features_engineering_output.csv")

X = df.drop("failure_in_next_24h", axis=1)
y = df["failure_in_next_24h"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

# Load models

log_model = joblib.load("model/baseline_logistic_gridsearch.joblib")
xgb_model = joblib.load("model/xgboost_tuned.joblib")

# Predict probabilities

log_prob = log_model.predict_proba(X_test)[:, 1]
xgb_prob = xgb_model.predict_proba(X_test)[:, 1]

# PR-AUC

log_pr_auc = average_precision_score(y_test, log_prob)
xgb_pr_auc = average_precision_score(y_test, xgb_prob)

# Threshold for XGBoost

precision, recall, thresholds = precision_recall_curve(y_test, xgb_prob)
PRECISION_TARGET = 0.90

valid = precision[:-1] >= PRECISION_TARGET
best_threshold = thresholds[valid][0] if valid.any() else 0.5

# Final predictions

log_pred = (log_prob >= 0.5).astype(int)
xgb_pred = (xgb_prob >= best_threshold).astype(int)

# Classification reports

log_report = classification_report(
    y_test, log_pred, output_dict=True
)
xgb_report = classification_report(
    y_test, xgb_pred, output_dict=True
)

print(log_report)
print(xgb_report)
# Convert to DataFrame

rows = []

for model_name, report, pr_auc in [
    ("Logistic Regression", log_report, log_pr_auc),
    ("XGBoost", xgb_report, xgb_pr_auc)
]:
    failure_metrics = report["1"]  # class 1 = failure

    rows.append({
        "model": model_name,
        "pr_auc": round(pr_auc, 4),
        "precision": round(failure_metrics["precision"], 4),
        "recall": round(failure_metrics["recall"], 4),
        "f1_score": round(failure_metrics["f1-score"], 4),
        "support": int(failure_metrics["support"])
    })

evaluation_df = pd.DataFrame(rows)

# Save CSV
evaluation_df.to_csv(
    "report/csv_report/model_evaluation_comparison.csv",
    index=False
)

print(" Model evaluation report saved as CSV")
print(evaluation_df)

# Plot
plt.figure()
plt.plot(log_recall, log_precision, label=f"Logistic Regression (PR-AUC={log_pr_auc:.2f})")
plt.plot(xgb_recall, xgb_precision, label=f"XGBoost (PR-AUC={xgb_pr_auc:.2f})")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision–Recall Curve Comparison")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("report/Image_report/pr_curve_comparison.png")
plt.show()