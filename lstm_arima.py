import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import date, timedelta
import streamlit as st
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf
import random
import math
import pmdarima as pm

# Ensure reproducibility by setting random seeds
np.random.seed(42)
tf.random.set_seed(42)
random.seed(42)

# Set a dark background style for better plot visualization
plt.style.use('dark_background')

# Streamlit app title and subtitle
st.title("StockViews")
st.subheader("Stock Trend Prediction with Hybrid Model (LSTM + ARIMA)")

# User inputs the stock ticker symbol
inp = st.text_input('Enter Stock Ticker', 'AAPL')

# Validate user input for the stock ticker
if not inp.strip():
    st.error("Error: The stock ticker cannot be blank. Please enter a valid stock ticker.")
    st.stop()

end = date.today()

# Cache the function to fetch stock data to improve performance
@st.cache_data
def fetch_data(ticker, end_date):
    return yf.download(ticker, end=end_date)

# Fetch historical stock data
data_all = fetch_data(inp, end)
if data_all.empty:
    st.error("Error: The stock ticker is incorrect or data is not available for the given ticker.")
    st.stop()

# Reset index and rename the adjusted close column
data_all.reset_index(inplace=True)
data_all.rename(columns={'Adj Close': 'Adj_Close'}, inplace=True)

# Determine the start date based on the data length
total_years = (data_all['Date'].iloc[-1] - data_all['Date'].iloc[0]).days / 365
if total_years > 5:
    start = end - timedelta(days=4*365)
else:
    start = end - timedelta(days=2*365)

# Filter the data for the desired date range
data = data_all[data_all['Date'] >= pd.to_datetime(start)]

# Ensure the dates are in datetime format and sorted
data['Date'] = pd.to_datetime(data['Date'])
data = data.sort_values(by=['Date'])

# Add the Previous_Close feature and drop rows with NaN values
data['Previous_Close'] = data['Close'].shift(1)
data.dropna(inplace=True)

# Define the function for ARIMA analysis with automatic parameter selection
def ARIMA_ALGO(df):
    df.set_index('Date', inplace=True)
    Quantity_date = df[['Close']]
    Quantity_date['Close'] = Quantity_date['Close'].astype(float)

    # Prepare training and testing data for the ARIMA model
    quantity = Quantity_date['Close'].values
    size = int(len(quantity) * 0.80)
    train, test = quantity[:size], quantity[size:]

    # Automatically select the best ARIMA model parameters
    model = pm.auto_arima(
        train, 
        seasonal=False, 
        stepwise=True, 
        suppress_warnings=True, 
        error_action='ignore', 
        trace=True 
    )
    
    print(f"Selected ARIMA Model: {model.summary()}")  # Print the model summary

    # Forecasting with the ARIMA model
    history = list(train)
    predictions = []
    for t in range(len(test)):
        model_fit = model.fit(history)  # Fit the model with the current history
        output = model_fit.predict(n_periods=1)  # Predict the next value
        yhat = output[0]
        predictions.append(yhat)
        history.append(test[t])  # Update history with the true value

    # Calculate error metrics for the ARIMA model
    mse = mean_squared_error(test, predictions)
    rmse = math.sqrt(mse)
    mae = mean_absolute_error(test, predictions)
    mape = np.mean(np.abs((test - predictions) / test)) * 100

    return predictions, test, mse, rmse, mae, mape

# Run ARIMA model and get predictions
arima_predictions, arima_test, arima_mse, arima_rmse, arima_mae, arima_mape = ARIMA_ALGO(data.copy())

# Select features for the LSTM model
features = ['Close', 'High', 'Low', 'Adj_Close', 'Volume', 'Previous_Close']

# Normalize the data for the LSTM model
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data[features])

# Create training and testing datasets
train_data_len = int(len(scaled_data) * 0.8)
train_data = scaled_data[:train_data_len]
test_data = scaled_data[train_data_len:]

# Function to create datasets with specified time steps
def create_dataset(dataset, time_step=1):
    X, y = [], []
    for i in range(len(dataset) - time_step):
        X.append(dataset[i:(i + time_step)])
        y.append(dataset[i + time_step, 0])  # Predicting the 'Close' price
    return np.array(X), np.array(y)

time_step = 60  
X_train, y_train = create_dataset(train_data, time_step)
X_test, y_test = create_dataset(test_data, time_step)

# Reshape input data for the LSTM model
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2])
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2])

# Build the LSTM model
def build_model():
    model = Sequential()
    model.add(LSTM(units=96, return_sequences=True, input_shape=(time_step, len(features))))
    model.add(LSTM(units=416, return_sequences=False))
    model.add(Dropout(rate=0.3))
    model.add(Dense(units=288))
    model.add(Dense(1))
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.002567442256450226), loss='mean_squared_error')
    return model

# Define early stopping to prevent overfitting
early_stop = EarlyStopping(monitor='val_loss', patience=10)

# Train the LSTM model
model = build_model()
history = model.fit(X_train, y_train, batch_size=32, epochs=200, verbose=1, validation_split=0.2, callbacks=[early_stop])

# Make predictions with the LSTM model
train_predict = model.predict(X_train)
test_predict = model.predict(X_test)

# Transform predictions back to the original scale
train_predict = scaler.inverse_transform(np.concatenate((train_predict, np.zeros((train_predict.shape[0], len(features) - 1))), axis=1))[:, 0]
test_predict = scaler.inverse_transform(np.concatenate((test_predict, np.zeros((test_predict.shape[0], len(features) - 1))), axis=1))[:, 0]

# Add predictions to the data
data['Train_Predict'] = np.nan
data['Test_Predict'] = np.nan

train_predict_len = len(train_predict)
test_predict_len = len(test_predict)

data.iloc[time_step:time_step + train_predict_len, data.columns.get_loc('Train_Predict')] = train_predict.flatten()
data.iloc[train_data_len + time_step:train_data_len + time_step + test_predict_len, data.columns.get_loc('Test_Predict')] = test_predict.flatten()

# Predict the latest data point with the LSTM model
latest_data = test_data[-time_step:]
latest_data = latest_data.reshape((1, time_step, len(features)))
lstm_pred = model.predict(latest_data)
lstm_pred = scaler.inverse_transform(np.concatenate((lstm_pred, np.zeros((lstm_pred.shape[0], len(features) - 1))), axis=1))[:, 0]

# Get the actual live price of the stock
live_data = yf.download(inp, period="1d", interval="1m")
actual_price = live_data['Close'].iloc[-1]

# Compare ARIMA and LSTM predictions to determine the best model
if abs(arima_predictions[-1] - actual_price) < abs(lstm_pred[0] - actual_price):
    best_model = "ARIMA"
    best_pred = arima_predictions[-1]
    data['Best_Predict'] = np.nan
    data.iloc[-len(arima_predictions):, data.columns.get_loc('Best_Predict')] = arima_predictions
else:
    best_model = "LSTM"
    best_pred = lstm_pred[0]
    data['Best_Predict'] = data['Test_Predict']

# Generate a buy/sell recommendation based on the predictions
recommendation = "Our prediction model shows that the predicted price of the stock is equal to the current actual price. This indicates that there is no expected change in the stock's value in the near future. In this case, we recommend you hold onto your stock, as there is no anticipated movement that would warrant buying or selling at this time. However, please note that prediction models aren't always accurate due to sudden market changes, so this is only a recommendation."
if best_pred > actual_price:
    recommendation = "Our prediction model indicates that the predicted price of the stock is higher than the current actual price. This suggests a potential increase in the stock's value. Based on this information, we recommend you consider buying the stock, as it is expected to appreciate, potentially offering you a profitable investment opportunity. However, please note that prediction models aren't always accurate due to sudden market changes, so this is only a recommendation."
elif best_pred < actual_price:
    recommendation = "According to our prediction model, the predicted price of the stock is lower than the current actual price. This implies a potential decrease in the stock's value. Therefore, we advise you to consider selling the stock to avoid potential losses. Selling now could help you preserve your capital and avoid a decrease in your investment's value. However, please note that prediction models aren't always accurate due to sudden market changes, so this is only a recommendation."

# Display the results
st.subheader(f"Best Model: {best_model}")
st.write(f"Predicted share price for {inp} today by {best_model} is ${best_pred:.2f}")
st.write(f"Actual live price for {inp} is ${actual_price:.2f}")

# Plot the closing price trend
st.subheader(f"Closing Price Trend for {inp}")
fig1, ax = plt.subplots(figsize=(12, 6))
ax.plot(data['Date'], data['Close'], label='Close', color='cyan')
ax.set_xlabel("Date")
ax.set_ylabel("Close Price")
ax.set_title("Closing Price", color='white')
ax.legend()
st.pyplot(fig1)

# Plot the predicted trend
st.subheader(f"Closing Price Predicted Trend for {inp}")
fig2, ax = plt.subplots(figsize=(12, 6))
ax.plot(data['Date'], data['Close'], label='Actual Price', color='cyan')
ax.plot(data['Date'], data['Train_Predict'], label='Train Predict', color='green', linestyle='--')
ax.plot(data['Date'], data['Test_Predict'], label='Test Predict', color='red', linestyle='--')
ax.set_xlabel("Date")
ax.set_ylabel("Price")
ax.set_title("Actual vs Predicted Closing Prices", color='white')
ax.legend()
st.pyplot(fig2)

# Plot a zoomed-in view of the last 30 days
st.subheader("Last 30 Days Prediction vs Actual")
fig3, ax3 = plt.subplots()
ax3.plot(data['Date'].iloc[-30:], data['Close'].iloc[-30:], label='Actual Close Price')
ax3.plot(data['Date'].iloc[-30:], data['Best_Predict'].iloc[-30:], label=f'{best_model} Predicted Close Price')
ax3.set_xlabel('Date')
ax3.set_ylabel('Price')
ax3.legend()
st.pyplot(fig3)

# Display the recommendation
st.write(f"Recommendation for {inp}: {recommendation}")
