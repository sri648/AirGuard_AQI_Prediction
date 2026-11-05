
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Load cleaned dataset
df = pd.read_csv("Data/cleaned_aqi.csv")

# Create output folder
os.makedirs("outputs", exist_ok=True)

print("========== AQI DATA ANALYSIS ==========")

print("\nDataset shape:")
print(df.shape)

print("\nAQI statistical summary:")
print(df["AQI"].describe())

print("\nPollution category counts:")
print(df["AQI_Bucket"].value_counts())

print("\nNumber of cities:")
print(df["City"].nunique())

# Chart 1: AQI category distribution
plt.figure(figsize=(9, 5))
sns.countplot(
    data=df,
    x="AQI_Bucket",
    order=df["AQI_Bucket"].value_counts().index
)
plt.title("AQI Pollution Category Distribution")
plt.xlabel("Pollution Category")
plt.ylabel("Number of Records")
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig("outputs/aqi_category_distribution.png")
plt.show()

# Chart 2: Top 10 cities by average AQI
city_aqi = (
    df.groupby("City")["AQI"]
    .mean()
    .sort_values(ascending=False)
    .head(10)
)

plt.figure(figsize=(10, 6))
sns.barplot(x=city_aqi.values, y=city_aqi.index)
plt.title("Top 10 Cities by Average AQI")
plt.xlabel("Average AQI")
plt.ylabel("City")
plt.tight_layout()
plt.savefig("outputs/top_10_cities_aqi.png")
plt.show()

print("\nEDA completed successfully!")
print("Charts saved inside outputs folder.")
