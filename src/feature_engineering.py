import pandas as pd
import joblib

SENSORS = ['temperature_c', 'vibration_rms_mm_s', 'pressure_bar']
WINDOWS = [1, 6, 12]

def feature_engineering(input_csv):
    df = pd.read_csv(input_csv)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values(['arm_id', 'timestamp'])

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
    
    df["failure_in_next_24h"] = (df["time_to_failure_hours"].notna() &(df["time_to_failure_hours"] <= 24)).astype(int)
    print(df["failure_in_next_24h"].value_counts())

    print(df.columns)
    print(df.isna().sum())
    print(df.info())
    print(df.describe())

    df.drop(["time_to_failure_hours","pressure_bar_roll_std_1h","vibration_rms_mm_s_roll_std_1h","temperature_c_roll_std_1h"], axis=1, inplace=True)
    print(df)

    return df


if __name__ == "__main__":
    df = feature_engineering("data/raw/factoryguard_synthetic_500.csv")

    df.to_csv("data/processed/features_engineering_output.csv.csv", index=False)
    joblib.dump(df, "data/processed/features_engineering.joblib")

    print("Feature engineering completed")
