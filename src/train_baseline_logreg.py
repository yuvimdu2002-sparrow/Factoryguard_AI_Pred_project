import joblib
import pandas as pd
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.linear_model import LogisticRegression

# Load engineered features (CSV)

df = pd.read_csv("data/processed/features_engineering_output.csv")
X = df.drop(["failure_in_next_24h"], axis=1)
y = df["failure_in_next_24h"]

feature_columns=X.columns.tolist()
print(feature_columns)
joblib.dump(feature_columns,"model/feature_columns.joblib")

print(df.columns)
print(df.nunique)
print(df.isna().sum())
print(df.info())
print(df.describe())
print(df.shape)
print(df["failure_in_next_24h"].value_counts())


X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2,stratify=y,random_state=42)

# Baseline Logistic Regression

log_reg = LogisticRegression(class_weight="balanced",max_iter=2000)

param_grid = {"C": [0.01, 0.1, 1, 10],"penalty": ["l2"]}

grid = GridSearchCV(log_reg, param_grid, scoring="average_precision",  cv=3, n_jobs=-1, verbose=1)

grid.fit(X_train, y_train)

best_model = grid.best_estimator_

print("Best parameters:", grid.best_params_)

joblib.dump(best_model,"model/baseline_logistic_gridsearch.joblib")

print(" Baseline Logistic Regression trained with GridSearch")
