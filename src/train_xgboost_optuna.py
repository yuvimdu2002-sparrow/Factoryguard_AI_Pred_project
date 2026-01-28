#import optuna
import joblib
import pandas as pd
#from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split

df = pd.read_csv("data/processed/features_engineering_output.csv")

X = df.drop(["failure_in_next_24h", "arm_id", "timestamp"], axis=1)
y = df["failure_in_next_24h"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
print(df.isna().sum())
'''
scale_pos_weight = y_train.value_counts()[0] / y_train.value_counts()[1]

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

    model.fit(X_train, y_train)
    return model.score(X_train, y_train)  # training-only

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=30)

best_model = XGBClassifier(
    **study.best_params,
    scale_pos_weight=scale_pos_weight,
    eval_metric="logloss",
    random_state=42
)

best_model.fit(X_train, y_train)

joblib.dump(best_model, "model/xgboost_tuned.joblib")
print(" XGBoost trained with Optuna")
'''