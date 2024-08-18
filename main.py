import pandas as pd
import streamlit as st
import datetime as dt
import joblib
import matplotlib.pyplot as plt

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

# Load the pre-trained SARIMA model
sarima_model = joblib.load('sarima_model.pkl')

# Sidebar options for user input
st.sidebar.header("Forecast Parameters")
future_steps = st.sidebar.number_input("Number of Days to Forecast", min_value=1, max_value=365, value=30)

# Plotting the historical data
st.subheader("Historical Data")
st.line_chart(df['meantemp'])

# Generate and plot the forecast
st.subheader(f"Forecast for the Next {future_steps} Days")

# Make predictions for the user-specified number of days
forecast = sarima_model.get_forecast(steps=future_steps)
forecast_index = pd.date_range(start=df.index[-1] + pd.Timedelta(days=1), periods=future_steps, freq='D')
forecast_mean = forecast.predicted_mean
forecast_series = pd.Series(forecast_mean, index=forecast_index)

# Plotting the forecast
plt.figure(figsize=(10, 6))
plt.plot(df.index, df['meantemp'], label='Historical Data')
plt.plot(forecast_series.index, forecast_series, label='Forecast')
plt.title(f"Temperature Forecast for the Next {future_steps} Days")
plt.xlabel("Date")
plt.ylabel("Temperature (°C)")
plt.legend()
st.pyplot(plt)

# Display forecasted values
st.write(f"Forecasted temperatures for the next {future_steps} days:")
st.dataframe(forecast_series)
