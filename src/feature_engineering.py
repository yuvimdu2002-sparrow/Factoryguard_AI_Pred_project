import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

df = pd.read_csv("/content/factoryguard_synthetic_500.csv")
df['timestamp'] = pd.to_datetime(df['timestamp'])
df = df.sort_values(['arm_id', 'timestamp'])

print(df.columns)
print(df.isna().sum())
print(df.info())
print(df.describe())

SENSORS = ['temperature_c', 'vibration_rms_mm_s', 'pressure_bar']
WINDOWS = [1, 6, 12]

for sensor in SENSORS:
    for w in WINDOWS:
        df[f'{sensor}_roll_mean_{w}h'] = (
            df.groupby('arm_id')[sensor]
              .transform(lambda x: x.rolling(w).mean())
        )

        df[f'{sensor}_roll_std_{w}h'] = (
            df.groupby('arm_id')[sensor]
              .transform(lambda y: y.rolling(w).std())
        )

        df[f'{sensor}_ema_{w}h'] = (
            df.groupby('arm_id')[sensor]
              .transform(lambda z: z.ewm(span=w, adjust=False).mean())
        )

    df[f'{sensor}_lag_1'] = df.groupby('arm_id')[sensor].shift(1)
    df[f'{sensor}_lag_2'] = df.groupby('arm_id')[sensor].shift(2)

df = df.groupby('arm_id').apply(lambda x: x.iloc[12:]).reset_index(drop=True)
df["failure_in_next_24h"]=(df["time_to_failure_hours"].notna() & (df["time_to_failure_hours"] <= 24)).astype(int)
print(df["failure_in_next_24h"].value_counts())
print(df.head())

print("Final shape:", df.shape)

df1 = df.drop(["time_to_failure_hours","pressure_bar_roll_std_1h","vibration_rms_mm_s_roll_std_1h","temperature_c_roll_std_1h"], axis=1)
print(df1)
print(df1.columns)
print(df1.isna().sum())
print(df1.info())
print(df1.describe())
corr=df1.corr(numeric_only=True)

plt.figure(figsize=(50,40))
sns.clustermap(corr, cmap='coolwarm')
plt.show()

joblib.dump(df, "factoryguard_feature_engineered.joblib")
df1.to_csv("feature_engineered_output.csv", index=False)
