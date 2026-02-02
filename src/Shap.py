import joblib
import shap
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("data/processed/features_engineering_output.csv")

X = df.drop(["failure_in_next_24h"], axis=1)

# Load FINAL production model
model = joblib.load("model/xgboost_tuned.joblib")

# SHAP Explainer (Tree-based)

explainer = shap.TreeExplainer(model)

background = X.sample(500, random_state=42)

shap_values = explainer.shap_values(background)

# GLOBAL EXPLANATION

plt.figure()
shap.summary_plot(shap_values, background, show=False)
plt.tight_layout()
plt.savefig("report/shap_global_feature_importance.png")
plt.close()

print("Global SHAP summary saved")

# LOCAL EXPLANATION (single failure case)
# Pick a row where model predicts high failure probability

failure_case = X.iloc[[0]]  # replace index with an actual failure if needed

# Compute SHAP values
shap_values = explainer(failure_case)

# Create explanation object properly
exp = shap.Explanation(
    values=shap_values.values[0],
    base_values=shap_values.base_values[0],
    data=failure_case.iloc[0],
    feature_names=X.columns
)

# Waterfall Plot
plt.figure(figsize=(10,6))
shap.plots.waterfall(exp, max_display=10, show=False)

# SAVE AFTER SHAP RENDER
plt.tight_layout()
plt.savefig("/content/drive/MyDrive/report/shap_local_failure_explanation.png", dpi=300, bbox_inches="tight")
plt.show()

print("Local SHAP explanation saved")