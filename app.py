
import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
from pathlib import Path

# ---------------------------------
# PAGE CONFIGURATION
# ---------------------------------

st.set_page_config(
    page_title="AirGuard | AQI Predictor",
    page_icon="🌍",
    layout="wide"
)

BASE_DIR = Path(__file__).parent
MODEL_DIR = BASE_DIR / "models"

# ---------------------------------
# LOAD MODELS AND DATA
# ---------------------------------

@st.cache_resource
def load_models():
    reg_model = joblib.load(MODEL_DIR / "aqi_regression_model.pkl")
    class_model = joblib.load(MODEL_DIR / "aqi_classification_model.pkl")
    feature_names = joblib.load(MODEL_DIR / "feature_names.pkl")
    metrics = joblib.load(MODEL_DIR / "model_metrics.pkl")
    return reg_model, class_model, feature_names, metrics

@st.cache_data
def load_data():
    return pd.read_csv(BASE_DIR / "Data" / "cleaned_aqi.csv")

reg_model, class_model, feature_names, metrics = load_models()
df = load_data()

# ---------------------------------
# HEADER
# ---------------------------------

st.title("🌍 AirGuard - AQI Prediction System")
st.markdown(
    "### Air Quality Index Prediction and Pollution Classification"
)
st.caption(
    "Machine Learning Based Air Quality Analysis | Team 2"
)

st.divider()

# ---------------------------------
# SIDEBAR
# ---------------------------------

st.sidebar.title("🌿 AirGuard")
st.sidebar.write("Enter pollution details to predict air quality.")

city = st.sidebar.selectbox(
    "Select City",
    sorted(df["City"].unique())
)

selected_date = st.sidebar.date_input(
    "Select Date",
    value=pd.Timestamp.today().date()
)

st.sidebar.subheader("Pollution Measurements")

pollutants = [
    "PM2.5", "PM10", "NO", "NO2", "NOx",
    "NH3", "CO", "SO2", "O3", "Benzene", "Toluene"
]

default_values = {
    "PM2.5": 50.0,
    "PM10": 80.0,
    "NO": 10.0,
    "NO2": 20.0,
    "NOx": 30.0,
    "NH3": 10.0,
    "CO": 1.0,
    "SO2": 10.0,
    "O3": 30.0,
    "Benzene": 2.0,
    "Toluene": 5.0
}

input_values = {}

for pollutant in pollutants:
    input_values[pollutant] = st.sidebar.number_input(
        pollutant,
        min_value=0.0,
        value=default_values[pollutant],
        step=1.0
    )

predict_button = st.sidebar.button(
    "🔍 Predict AQI",
    use_container_width=True
)

# ---------------------------------
# MAIN DASHBOARD
# ---------------------------------

st.subheader("📊 Model Performance")

col1, col2, col3 = st.columns(3)

col1.metric("Regression R²", f"{metrics['R2 Score'] * 100:.2f}%")
col2.metric("Classification Accuracy",
            f"{metrics['Classification Accuracy'] * 100:.2f}%")
col3.metric("Regression MAE", f"{metrics['MAE']:.2f}")

st.divider()

# ---------------------------------
# PREDICTION
# ---------------------------------

if predict_button:

    input_data = {
        **input_values,
        "Year": selected_date.year,
        "Month": selected_date.month,
        "Day": selected_date.day,
        "DayOfWeek": selected_date.weekday()
    }

    input_df = pd.DataFrame([input_data])

    # Add city one-hot encoded columns
    for column in feature_names:
        if column.startswith("City_"):
            input_df[column] = int(column == f"City_{city}")

    # Arrange features in the exact training order
    input_df = input_df.reindex(
        columns=feature_names,
        fill_value=0
    )

    predicted_aqi = reg_model.predict(input_df)[0]
    predicted_category = class_model.predict(input_df)[0]

    st.subheader("🌍 AQI Prediction Result")

    result1, result2 = st.columns(2)

    result1.metric(
        "Predicted AQI",
        f"{predicted_aqi:.2f}"
    )

    result2.metric(
        "Predicted Pollution Category",
        predicted_category
    )

    st.success("AQI prediction completed successfully!")

    # Show input pollution values
    st.subheader("Pollution Measurement Analysis")

    chart_data = pd.DataFrame({
        "Pollutant": list(input_values.keys()),
        "Value": list(input_values.values())
    })

    fig = px.bar(
        chart_data,
        x="Pollutant",
        y="Value",
        title="Entered Pollution Measurements",
        color="Value",
        color_continuous_scale="Viridis"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.info(
        "This is a machine-learning estimate based on the values entered. "
        "It is not a live sensor reading or official AQI measurement."
    )

else:
    st.info(
        "👈 Select a city, date and pollution measurements, "
        "then click Predict AQI."
    )

# ---------------------------------
# DATASET OVERVIEW
# ---------------------------------

st.divider()
st.subheader("📁 Dataset Overview")

c1, c2, c3 = st.columns(3)

c1.metric("Total Records", f"{len(df):,}")
c2.metric("Cities Covered", df["City"].nunique())
c3.metric("Pollution Categories", df["AQI_Bucket"].nunique())

st.subheader("Pollution Category Distribution")

category_data = (
    df["AQI_Bucket"].value_counts().reset_index()
)
category_data.columns = ["Category", "Records"]

fig2 = px.pie(
    category_data,
    names="Category",
    values="Records",
    title="AQI Category Distribution",
    hole=0.4
)

st.plotly_chart(fig2, use_container_width=True)

st.caption("Developed as a supervised machine learning major project | Team 2")
