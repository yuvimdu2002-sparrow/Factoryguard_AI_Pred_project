import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    classification_report
)

# Load processed data
df = pd.read_csv("data/processed/features_engineering_output.csv")
print(df)

X = df.drop(["failure_in_next_24h"], axis=1)
y = df["failure_in_next_24h"]

# Train / Test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

# Load FINAL production model
loaded = joblib.load("model/xgboost_tuned.joblib")

if isinstance(loaded, dict):
    model = loaded["model"]
    saved_threshold = loaded.get("threshold", None)
else:
    model = loaded
    saved_threshold = None

# Predict on TEST data only

y_prob_test = model.predict_proba(X_test)[:, 1]

# PR-AUC on TEST set (MANDATORY)

pr_auc = average_precision_score(y_test, y_prob_test)
print(f"PR-AUC (Test Set): {pr_auc:.4f}")

# High-precision threshold
# (use saved threshold if available,
# otherwise compute from TEST set)

if saved_threshold is not None:
    best_threshold = saved_threshold
else:
    precision, recall, thresholds = precision_recall_curve(y_test, y_prob_test)
    PRECISION_TARGET = 0.90
    valid = precision[:-1] >= PRECISION_TARGET
    best_threshold = thresholds[valid][0] if valid.any() else 0.5

print("Chosen threshold:", best_threshold)

# Final TEST predictions

y_pred_test = (y_prob_test >= best_threshold).astype(int)

print("\nFinal Evaluation on TEST set (High Precision):")
print(classification_report(y_test, y_pred_test, digits=4))
