import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv("data/raw/factoryguard_synthetic_500.csv")

# Convert 'timestamp' to datetime objects for proper plotting
df['timestamp'] = pd.to_datetime(df['timestamp'])

# Set the style for the plots
sns.set_style("whitegrid")

print("Generating visualizations for the DataFrame...")


# 1. Histogram for 'temperature_c'
plt.figure(figsize=(10, 6))
sns.histplot(df['temperature_c'], kde=True)
plt.title('Distribution of Temperature (Celsius)')
plt.xlabel('Temperature (C)')
plt.ylabel('Frequency')
plt.show()

# 2. Time-series plot for 'temperature_c' over time
plt.figure(figsize=(14, 7))
sns.lineplot(data=df, x='timestamp', y='temperature_c', hue='arm_id')
plt.title('Temperature (Celsius) Over Time per Arm')
plt.xlabel('Timestamp')
plt.ylabel('Temperature (C)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 3. Time-series plot for 'vibration_rms_mm_s' over time
plt.figure(figsize=(14, 7))
sns.lineplot(data=df, x='timestamp', y='vibration_rms_mm_s', hue='arm_id')
plt.title('Vibration RMS (mm/s) Over Time per Arm')
plt.xlabel('Timestamp')
plt.ylabel('Vibration RMS (mm/s)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

print("Visualizations generated successfully.")

sns.pairplot(df)
plt.show()

sns.heatmap(df.corr(numeric_only=True), annot=True)
plt.show()

sns.scatterplot(data=df, x='temperature_c', y='vibration_rms_mm_s')
plt.show()

sns.boxplot(data=df, x='arm_id', y='temperature_c')
plt.show()

sns.histplot(data=df, x='temperature_c', hue='arm_id', multiple='stack')
plt.show()