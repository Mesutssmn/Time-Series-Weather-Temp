import pandas as pd
import numpy as np
import streamlit as st
import datetime as dt
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Load the dataset
@st.cache_data
def load_data():
    train = pd.read_csv('ClimateTrain.csv')
    test = pd.read_csv('ClimateTest.csv')
    df = pd.concat([train, test])
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    df = df["meantemp"].resample('W').mean()  # Resampling to weekly mean
    return df

# Load and process the dataset
df = load_data()

# Sidebar options
st.sidebar.header("Select Forecast Model")
forecast_type = st.sidebar.selectbox("Choose a Model", ["Triple Exponential Smoothing (Holt-Winters)", "SARIMA"])

# User-defined future prediction steps (in weeks)
st.sidebar.header("Forecasting Parameters")
future_steps = st.sidebar.number_input("Number of Weeks to Forecast", min_value=1, max_value=52, value=12)

# Plotting the historical data
st.subheader("Historical Data")
st.line_chart(df)

# Function to display forecast results
def plot_forecast(historical, forecast, title):
    plt.figure(figsize=(10, 6))
    plt.plot(historical.index, historical, label='Historical Data')
    plt.plot(forecast.index, forecast, label='Forecast')
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Temperature (°C)")
    plt.legend()
    st.pyplot(plt)

# Model selection and forecasting
try:
    if forecast_type == "Triple Exponential Smoothing (Holt-Winters)":
        st.subheader(f"Forecasting the Next {future_steps} Weeks using Triple Exponential Smoothing (Holt-Winters)")
        alpha, beta, gamma = 0.3, 0.5, 0.4  # Pre-optimized parameters
        # Adjust the seasonal_periods for weekly data
        tes_model = ExponentialSmoothing(df, trend="add", seasonal="add", seasonal_periods=52).fit(
            smoothing_level=alpha, smoothing_trend=beta, smoothing_seasonal=gamma)
        forecast_index = pd.date_range(start=df.index[-1] + pd.Timedelta(weeks=1), periods=future_steps, freq='W')
        forecast = pd.Series(tes_model.forecast(future_steps), index=forecast_index)
        if forecast.isnull().values.any():
            raise ValueError("The Triple Exponential Smoothing model returned missing values.")
        plot_forecast(df, forecast, "Triple Exponential Smoothing Forecast")
        st.write(f"Forecasted temperatures for the next {future_steps} weeks:")
        st.dataframe(forecast)

    elif forecast_type == "SARIMA":
        st.subheader(f"Forecasting the Next {future_steps} Weeks using SARIMA")
        order = (1, 0, 1)
        seasonal_order = (1, 1, 1, 52)  # Pre-optimized parameters for weekly data
        sarima_model = SARIMAX(df, order=order, seasonal_order=seasonal_order).fit()
        forecast_index = pd.date_range(start=df.index[-1] + pd.Timedelta(weeks=1), periods=future_steps, freq='W')
        forecast = pd.Series(sarima_model.get_forecast(steps=future_steps).predicted_mean, index=forecast_index)
        if forecast.isnull().values.any():
            raise ValueError("The SARIMA model returned missing values.")
        plot_forecast(df, forecast, "SARIMA Forecast")
        st.write(f"Forecasted temperatures for the next {future_steps} weeks:")
        st.dataframe(forecast)

except ValueError as e:
    st.error(f"Model Error: {e}")
    st.write("Please try a different model or adjust the model parameters.")
