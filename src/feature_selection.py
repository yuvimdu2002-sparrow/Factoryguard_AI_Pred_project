import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif, RFE
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv("data/raw/factoryguard_synthetic_500.csv")
report=pd.DataFrame()

# Convert 'timestamp' to datetime objects for proper plotting
df['timestamp'] = pd.to_datetime(df['timestamp'])

df["failure_in_next_24h"] = (df["time_to_failure_hours"].notna() &(df["time_to_failure_hours"] <= 24)).astype(int)
    

X = df.drop(["failure_in_next_24h", "arm_id", "timestamp","time_to_failure_hours"], axis=1)
y = df["failure_in_next_24h"]
report=pd.DataFrame()
report["Feauture"]=X.columns

#Embedded Method
rf = RandomForestClassifier(class_weight="balanced",random_state=42,n_estimators=200)
rf.fit(X,y)
rf_importance=pd.DataFrame({'Feature':X.columns,'Importance':rf.feature_importances_}).sort_values(by='Importance',ascending=False)
print(rf_importance.head(10))


top_feature=rf_importance['Feature'].head(5).tolist()
print("Top Feautures:","".join(top_feature))
report["Embedded Method Importance"]=rf.feature_importances_

#filter method
fea = SelectKBest(score_func=f_classif,k=7)
X_selected=fea.fit_transform(X,y)

feature_score=pd.DataFrame({'Feature':X.columns,'Score':fea.scores_,'P_values':fea.pvalues_}).sort_values(by='Score',ascending=False)
print(feature_score)
best_features=X.columns[fea.get_support()]
print(best_features)
report["filter_method_Score"]=fea.scores_
report["filter_method_P_Value"]=fea.pvalues_

#wrapper method
model = LogisticRegression(class_weight="balanced",max_iter=2000,n_jobs=-1)

rfe=RFE(estimator=model, n_features_to_select=6)
rfe.fit(X,y)
rfe_rank=pd.DataFrame({'Feature':X.columns,'Rank':rfe.ranking_}).sort_values(by='Rank')
print(rfe_rank)

best_feature=X.columns[rfe.support_]

print("best features",best_feature)
#final report for feature selection
report['Wrap_method_Rank']=rfe.ranking_
print(report)
report.to_csv("feature_selection_report.csv", index=False)

