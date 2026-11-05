
import pandas as pd

print("Loading AQI dataset...")

# Load original dataset
df = pd.read_csv("Data/city_day.csv")

print("Original dataset shape:", df.shape)

# Remove rows where prediction targets are missing
df = df.dropna(subset=["AQI", "AQI_Bucket"])

# Remove duplicate records
df = df.drop_duplicates()

# Remove Xylene because it has too many missing values
df = df.drop(columns=["Xylene"])

# Select pollution columns
pollutant_columns = [
    "PM2.5", "PM10", "NO", "NO2", "NOx",
    "NH3", "CO", "SO2", "O3", "Benzene", "Toluene"
]

# Fill missing values using city-wise median
for column in pollutant_columns:
    df[column] = df.groupby("City")[column].transform(
        lambda values: values.fillna(values.median())
    )

    # Use overall median if any values are still missing
    df[column] = df[column].fillna(df[column].median())

# Convert Date into datetime format
df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

# Remove rows with invalid dates
df = df.dropna(subset=["Date"])

# Save cleaned dataset
df.to_csv("Data/cleaned_aqi.csv", index=False)

print("\n========== CLEANING COMPLETED ==========")
print("Cleaned dataset shape:", df.shape)

print("\nRemaining missing values:")
print(df.isnull().sum())

print("\nCleaned dataset saved successfully!")
print("File: Data/cleaned_aqi.csv")
