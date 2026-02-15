# ==========================================
# Simple Tractor Forecast App
# ==========================================

import streamlit as st
import pandas as pd
import pickle
import os

# ------------------------------------------
# Page Configuration
# ------------------------------------------
st.set_page_config(
    page_title="Tractor Forecast",
    layout="centered"
)

st.title("Tractor Sales Forecast")
st.write("Forecast Between 2014 – 2025")

# ------------------------------------------
# Load Data
# ------------------------------------------
@st.cache_data
def load_data():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(base_dir, "Tractor-Sales.csv")

    df = pd.read_csv(file_path)
    df['Month-Year'] = pd.to_datetime(df['Month-Year'], format='%b-%y')
    df = df.set_index('Month-Year')
    return df

# ------------------------------------------
# Load Model
# ------------------------------------------
@st.cache_resource
def load_model():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, "exponential_smoothing_model.pkl")

    with open(model_path, "rb") as file:
        model = pickle.load(file)

    return model

df = load_data()
model = load_model()

# ------------------------------------------
# Month & Year Selection
# ------------------------------------------
months = [
    "January","February","March","April","May","June",
    "July","August","September","October","November","December"
]

selected_month = st.selectbox("Select Month", months)
selected_year = st.number_input("Select Year", min_value=2014, max_value=2025, value=2022)

# ------------------------------------------
# Prediction Logic
# ------------------------------------------
if st.button("Get Result"):

    selected_date = pd.to_datetime(f"01-{selected_month}-{selected_year}")

    # If date exists in historical data
    if selected_date in df.index:
        actual_value = df.loc[selected_date]["Number of Tractor Sold"]
        st.subheader("Historical Sales")
        st.write(f"Sales for {selected_month} {selected_year}:")
        st.success(f"{round(actual_value)} Units")

    else:
        last_training_date = df.index[-1]

        months_diff = (selected_date.year - last_training_date.year) * 12 + \
                      (selected_date.month - last_training_date.month)

        if months_diff > 0:
            forecast = model.forecast(months_diff)
            predicted_value = forecast.iloc[-1]

            st.subheader("Forecasted Sales")
            st.write(f"Predicted Sales for {selected_month} {selected_year}:")
            st.success(f"{round(predicted_value)} Units")
        else:
            st.warning("Data not available for selected period.")

# ------------------------------------------
# Footer
# ------------------------------------------
st.markdown("---")
st.caption("Tractor Forecast Dashboard | 2014 – 2025")
