import pandas as pd
import joblib

df = pd.read_csv("data/raw/factoryguard_synthetic_500.csv")
df['timestamp'] = pd.to_datetime(df['timestamp'])
df = df.sort_values(['arm_id', 'timestamp'])
df["failure_in_next_24h"] = (df["time_to_failure_hours"].notna() &(df["time_to_failure_hours"] <= 24)).astype(int)
print(df["failure_in_next_24h"].value_counts())
    
def feature_engineering(SENSORS,WINDOWS):
    global df
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
                  .transform(lambda x: x.ewm(span=w, adjust=False).mean())
            )

        df[f'{sensor}_lag_1'] = df.groupby('arm_id')[sensor].shift(1)
        df[f'{sensor}_lag_2'] = df.groupby('arm_id')[sensor].shift(2)
        

    df = df.groupby('arm_id').apply(lambda x: x.iloc[12:]).reset_index(drop=True)
    
    return df



if __name__ == "__main__":
    feature_engineering(['temperature_c', 'vibration_rms_mm_s', 'pressure_bar'],[1, 6, 12])
    joblib.dump(df, "model/features_engineering.joblib")

    # drop columns with the help of feature selection report
    drop_features=["arm_id","timestamp","time_to_failure_hours","pressure_bar_roll_std_1h","vibration_rms_mm_s_roll_std_1h","rpm",
                "temperature_c_roll_std_1h","error_count","load_pct","maintenance_days_ago","humidity_pct","age_hours"]
    df.drop(drop_features, axis=1, inplace=True)
    print(df.columns)
    print(df.isna().sum())
    print(df.info())
    print(df.describe())

    df.to_csv("data/processed/features_engineering_output.csv", index=False)
    
    print("Feature engineering completed")
