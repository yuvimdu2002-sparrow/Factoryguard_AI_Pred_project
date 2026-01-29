import joblib
import shap
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("/content/drive/MyDrive/Zaalima project/features_engineering_output.csv")

X = df.drop(["failure_in_next_24h","arm_id","timestamp"], axis=1)

# Load FINAL production model
model = joblib.load("/content/drive/MyDrive/Zaalima project/xgboost_tuned.joblib")

# SHAP Explainer (Tree-based)

explainer = shap.TreeExplainer(model)

background = X.sample(500, random_state=42)

shap_values = explainer.shap_values(background)

# GLOBAL EXPLANATION

plt.figure()
shap.summary_plot(shap_values, background, show=False)
plt.tight_layout()
plt.savefig("/content/drive/MyDrive/Zaalima project/shap_global_feature_importance.png")
plt.close()

print("Global SHAP summary saved")

# LOCAL EXPLANATION (single failure case)
# Pick a row where model predicts high failure probability

failure_case = X.iloc[[0]]  # replace index with an actual failure if needed

shap_value_single = explainer.shap_values(failure_case)

shap.force_plot(
    explainer.expected_value,
    shap_value_single,
    failure_case,
    matplotlib=True,
    show=False
)

plt.savefig("/content/drive/MyDrive/Zaalima project/shap_local_failure_explanation.png")
plt.close()

print("Local SHAP explanation saved")