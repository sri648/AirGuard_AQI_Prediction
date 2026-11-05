
import pandas as pd
import joblib
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
    classification_report
)

BASE_DIR = Path(__file__).parent
DATA_PATH = BASE_DIR / "Data" / "cleaned_aqi.csv"
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)

print("Loading cleaned AQI dataset...")
df = pd.read_csv(DATA_PATH)

# Date preprocessing
df["Date"] = pd.to_datetime(df["Date"])
df["Year"] = df["Date"].dt.year
df["Month"] = df["Date"].dt.month
df["Day"] = df["Date"].dt.day
df["DayOfWeek"] = df["Date"].dt.day_of_week

# Input features
features = [
    "City", "PM2.5", "PM10", "NO", "NO2", "NOx",
    "NH3", "CO", "SO2", "O3", "Benzene", "Toluene",
    "Year", "Month", "Day", "DayOfWeek"
]

X = pd.get_dummies(df[features], columns=["City"], dtype=int)
feature_names = X.columns.tolist()

# ====================================
# MODEL 1: AQI REGRESSION
# ====================================

print("\nTraining AQI Regression Model...")

y_reg = df["AQI"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y_reg, test_size=0.2, random_state=42
)

reg_model = RandomForestRegressor(
    n_estimators=60,
    max_depth=18,
    max_leaf_nodes=1000,
    min_samples_leaf=3,
    random_state=42,
    n_jobs=-1
)

reg_model.fit(X_train, y_train)
reg_predictions = reg_model.predict(X_test)

mae = mean_absolute_error(y_test, reg_predictions)
rmse = mean_squared_error(y_test, reg_predictions) ** 0.5
r2 = r2_score(y_test, reg_predictions)

print("\nAQI REGRESSION RESULTS")
print("MAE:", round(mae, 2))
print("RMSE:", round(rmse, 2))
print("R2 Score:", round(r2, 4))

# ====================================
# MODEL 2: POLLUTION CLASSIFICATION
# ====================================

print("\nTraining Pollution Classification Model...")

y_class = df["AQI_Bucket"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y_class, test_size=0.2,
    random_state=42, stratify=y_class
)

class_model = RandomForestClassifier(
    n_estimators=60,
    max_depth=18,
    max_leaf_nodes=1000,
    min_samples_leaf=3,
    random_state=42,
    n_jobs=-1,
    class_weight="balanced"
)

class_model.fit(X_train, y_train)
class_predictions = class_model.predict(X_test)

accuracy = accuracy_score(y_test, class_predictions)

print("\nCLASSIFICATION RESULTS")
print("Accuracy:", round(accuracy * 100, 2), "%")
print("\nClassification Report:")
print(classification_report(y_test, class_predictions))

# ====================================
# SAVE COMPRESSED MODELS
# ====================================

joblib.dump(
    reg_model,
    MODEL_DIR / "aqi_regression_model.pkl",
    compress=3
)

joblib.dump(
    class_model,
    MODEL_DIR / "aqi_classification_model.pkl",
    compress=3
)

joblib.dump(
    feature_names,
    MODEL_DIR / "feature_names.pkl",
    compress=3
)

metrics = {
    "MAE": mae,
    "RMSE": rmse,
    "R2 Score": r2,
    "Classification Accuracy": accuracy
}

joblib.dump(
    metrics,
    MODEL_DIR / "model_metrics.pkl",
    compress=3
)

print("\n================================")
print("BOTH MODELS TRAINED SUCCESSFULLY!")
print("Compressed models saved successfully.")
print("================================")
