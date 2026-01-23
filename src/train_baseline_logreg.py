import joblib
import pandas as pd
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.linear_model import LogisticRegression

# Load engineered features (CSV)
df = pd.read_csv("/content/drive/MyDrive/Zaalima project/features_engineering_output.csv")
X = df.drop(["failure_in_next_24h", "arm_id", "timestamp"], axis=1)
y = df["failure_in_next_24h"]

X_train, _, y_train, _ = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

# Baseline Logistic Regression

log_reg = LogisticRegression(
    class_weight="balanced",
    max_iter=2000,
    n_jobs=-1
)

param_grid = {
    "C": [0.01, 0.1, 1, 10],
    "penalty": ["l2"]
}

grid = GridSearchCV(
    log_reg,
    param_grid,
    scoring="average_precision",  # PR-AUC
    cv=3,
    n_jobs=-1,
    verbose=1
)

grid.fit(X_train, y_train)

best_model = grid.best_estimator_

print("Best parameters:", grid.best_params_)

joblib.dump(
    best_model,
    "/content/drive/MyDrive/Zaalima project/baseline_logistic_gridsearch.joblib"
)

print(" Baseline Logistic Regression trained with GridSearch")
