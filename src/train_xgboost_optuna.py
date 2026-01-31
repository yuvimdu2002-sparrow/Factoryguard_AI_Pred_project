import optuna
import joblib
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, average_precision_score

#  LOAD DATA
df = pd.read_csv("/content/drive/MyDrive/Zaalima project/features_engineering_output.csv")

X = df.drop(["failure_in_next_24h"], axis=1)
y = df["failure_in_next_24h"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

scale_pos_weight = y_train.value_counts()[0] / y_train.value_counts()[1]

# CROSS VALIDATION
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# OPTUNA OBJECTIVE

def objective(trial):
    model = XGBClassifier(
        n_estimators=trial.suggest_int("n_estimators", 200, 600),
        max_depth=trial.suggest_int("max_depth", 3, 8),
        learning_rate=trial.suggest_float("learning_rate", 0.01, 0.1),
        subsample=trial.suggest_float("subsample", 0.6, 1.0),
        colsample_bytree=trial.suggest_float("colsample_bytree", 0.6, 1.0),
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
        random_state=42
    )

    scores = cross_val_score(
        model,
        X_train,
        y_train,
        cv=cv,
        scoring="average_precision"   # PR-AUC
    )

    return scores.mean()

# OPTUNA SEARCH
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=30)

# FINAL MODEL
best_model = XGBClassifier(
    **study.best_params,
    scale_pos_weight=scale_pos_weight,
    eval_metric="logloss",
    random_state=42
)

best_model.fit(X_train, y_train)

#  EVALUATION
y_pred = best_model.predict(X_test)
y_probs = best_model.predict_proba(X_test)[:, 1]

print("Accuracy :", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("PR-AUC   :", average_precision_score(y_test, y_probs))

# SAVE MODEL
joblib.dump(best_model, "/content/drive/MyDrive/Zaalima project/xgboost_tuned.joblib")

print("XGBoost trained with Optuna + Cross Validation")
