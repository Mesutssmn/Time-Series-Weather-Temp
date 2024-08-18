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
    return df

# Load and process the dataset
df = load_data()

# Sidebar options
st.sidebar.header("Select Forecast Type")
forecast_type = st.sidebar.selectbox("Select Model", ["Single Exponential Smoothing", "Double Exponential Smoothing", "Triple Exponential Smoothing", "SARIMA"])

# User-defined date range
st.sidebar.header("Select Date Range")
start_date = st.sidebar.date_input("Start Date", dt.date(2013, 1, 1))
end_date = st.sidebar.date_input("End Date", dt.datetime.today().date())

# Filter data based on the user-selected date range
filtered_data = df.loc[start_date:end_date]

# Check if the filtered data is empty
if filtered_data.empty:
    st.error("No data available for the selected date range. Please choose a different range.")
else:
    # Plotting the data
    st.line_chart(filtered_data['meantemp'])

    # Function to display forecast results
    def plot_forecast(train, test, y_pred, title):
        plt.figure(figsize=(10, 6))
        plt.plot(train.index, train, label='Train')
        plt.plot(test.index, test, label='Test')
        plt.plot(test.index, y_pred, label='Forecast')
        plt.title(title)
        plt.legend()
        st.pyplot(plt)

    # Train-test split
    train = filtered_data['meantemp'][:int(len(filtered_data)*0.8)]
    test = filtered_data['meantemp'][int(len(filtered_data)*0.8):]

    # Single Exponential Smoothing
    if forecast_type == "Single Exponential Smoothing":
        st.subheader("Single Exponential Smoothing")
        alpha = 0.1  # Best alpha after optimization
        ses_model = ExponentialSmoothing(train).fit(smoothing_level=alpha)
        y_pred_ses = ses_model.forecast(len(test))
        plot_forecast(train, test, y_pred_ses, "Single Exponential Smoothing Forecast")
        st.write(f"Mean Absolute Error (MAE): {np.mean(np.abs(test - y_pred_ses)):.4f}")

    # Double Exponential Smoothing
    elif forecast_type == "Double Exponential Smoothing":
        st.subheader("Double Exponential Smoothing")
        alpha, beta = 0.1, 0.3  # Best parameters after optimization
        des_model = ExponentialSmoothing(train, trend="add").fit(smoothing_level=alpha, smoothing_slope=beta)
        y_pred_des = des_model.forecast(len(test))
        plot_forecast(train, test, y_pred_des, "Double Exponential Smoothing Forecast")
        st.write(f"Mean Absolute Error (MAE): {np.mean(np.abs(test - y_pred_des)):.4f}")

    # Triple Exponential Smoothing (Holt-Winters)
    elif forecast_type == "Triple Exponential Smoothing":
        st.subheader("Triple Exponential Smoothing (Holt-Winters)")
        alpha, beta, gamma = 0.3, 0.5, 0.4  # Best parameters after optimization
        tes_model = ExponentialSmoothing(train, trend="add", seasonal="add", seasonal_periods=52).fit(smoothing_level=alpha, smoothing_slope=beta, smoothing_seasonal=gamma)
        y_pred_tes = tes_model.forecast(len(test))
        plot_forecast(train, test, y_pred_tes, "Triple Exponential Smoothing Forecast")
        st.write(f"Mean Absolute Error (MAE): {np.mean(np.abs(test - y_pred_tes)):.4f}")

    # SARIMA
    elif forecast_type == "SARIMA":
        st.subheader("SARIMA")
        order = (1, 1, 1)  # Order parameters
        seasonal_order = (1, 1, 1, 52)  # Seasonal order parameters
        sarima_model = SARIMAX(train, order=order, seasonal_order=seasonal_order).fit()
        y_pred_sarima = sarima_model.get_forecast(steps=len(test)).predicted_mean
        plot_forecast(train, test, y_pred_sarima, "SARIMA Forecast")
        st.write(f"Mean Absolute Error (MAE): {np.mean(np.abs(test - y_pred_sarima)):.4f}")
