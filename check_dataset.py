import pandas as pd

# Load the AQI dataset
df = pd.read_csv("Data/city_day.csv")

# Display dataset information
print("\n========== DATASET INFORMATION ==========")

print("\nDataset Shape:")
print(df.shape)

print("\nColumn Names:")
print(df.columns.tolist())

print("\nFirst 5 Records:")
print(df.head())

print("\nMissing Values:")
print(df.isnull().sum())

print("\nDataset Information:")
df.info()