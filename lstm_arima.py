import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import date, timedelta, datetime
import streamlit as st
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf
import random
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error
import math
import os

# Set seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)
random.seed(42)

# Applying a dark background style
plt.style.use('dark_background')

st.title("Stock Trend Prediction with Hybrid Model (LSTM + ARIMA)")
#asks the user to input stock ticker
inp = st.text_input('Enter Stock Ticker', 'AAPL')  
start = '2022-05-10'
end = date.today()

# Fetching all data using yfinance 
data = yf.download(inp, start=start, end=end)
data.reset_index(inplace=True)

# Ensure columns are named correctly 
data.rename(columns={'Adj Close': 'Adj_Close'}, inplace=True) 

# Preparing data and checking if in date time, and sorting date
data['Date'] = pd.to_datetime(data['Date'])
data = data.sort_values(by=['Date'])

# ARIMA analysis function
def ARIMA_ALGO(df, quote):
    df['Date'] = pd.to_datetime(df.index)
    Quantity_date = df[['Close', 'Date']]
    Quantity_date.index = Quantity_date['Date']
    Quantity_date['Close'] = Quantity_date['Close'].astype(float)
    Quantity_date = Quantity_date.drop(['Date'], axis=1)

    # Prepare training and testing data for the ARIMA model
    quantity = Quantity_date['Close'].values
    size = int(len(quantity) * 0.80)
    train, test = quantity[0:size], quantity[size:]

    # ARIMA model
    history = list(train)
    predictions = []
    for t in range(len(test)):
        model = ARIMA(history, order=(1, 0, 1))
        model_fit = model.fit()
        output = model_fit.forecast()
        yhat = output[0]
        predictions.append(yhat)
        history.append(test[t])

    # Calculate error metrics
    rmse = math.sqrt(mean_squared_error(test, predictions))
    mae = mean_absolute_error(test, predictions)
    mape = np.mean(np.abs((test - predictions) / test)) * 100

    return predictions, test, rmse, mae, mape

# Running ARIMA model
arima_predictions, arima_test, arima_rmse, arima_mae, arima_mape = ARIMA_ALGO(data, inp)

# Normalizing the data for LSTM
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data[['Close']])

# Creating the training and testing dataset
train_data_len = int(len(scaled_data) * 0.8)
train_data = scaled_data[:train_data_len]
test_data = scaled_data[train_data_len:]

def create_dataset(dataset, time_step=1):
    X, y = [], []
    for i in range(len(dataset) - time_step):
        a = dataset[i:(i + time_step), 0]
        X.append(a)
        y.append(dataset[i + time_step, 0])
    return np.array(X), np.array(y)

time_step = 60  
X_train, y_train = create_dataset(train_data, time_step)
X_test, y_test = create_dataset(test_data, time_step)

# Reshape input
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# Build the LSTM model with specified hyperparameters (these were chosen after tuning hypermeters)
def build_model():
    model = Sequential()
    model.add(LSTM(units=416, return_sequences=True, input_shape=(time_step, 1)))
    model.add(LSTM(units=352, return_sequences=False))
    model.add(Dropout(rate=0.2))
    model.add(Dense(units=112))
    model.add(Dense(1))
   
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mean_squared_error')
    return model

# Define early stopping
early_stop = EarlyStopping(monitor='val_loss', patience=10)

# Train the model with the specified hyperparameters
model = build_model()
history = model.fit(X_train, y_train, batch_size=32, epochs=200, verbose=1, validation_split=0.2, callbacks=[early_stop])

# Predicting with LSTM
train_predict = model.predict(X_train)
test_predict = model.predict(X_test)

# Transform back to original form
train_predict = scaler.inverse_transform(train_predict)
test_predict = scaler.inverse_transform(test_predict)

# Ensure test data is aligned with predictions for LSTM
test_data_aligned = test_data[time_step:].flatten()  

# Adding predictions to the data
data['Train_Predict'] = np.nan
data['Test_Predict'] = np.nan

# Ensure the lengths match when inserting predictions
train_predict_len = len(train_predict)
test_predict_len = len(test_predict)

data.iloc[time_step:time_step + train_predict_len, data.columns.get_loc('Train_Predict')] = train_predict.flatten()
data.iloc[train_data_len + time_step:train_data_len + time_step + test_predict_len, data.columns.get_loc('Test_Predict')] = test_predict.flatten()

# Running LSTM model
latest_data = test_data[-time_step:]
latest_data = latest_data.reshape((1, time_step, 1))
lstm_pred = model.predict(latest_data)
lstm_pred = scaler.inverse_transform(lstm_pred)

# Calculate metrics for LSTM
lstm_rmse = math.sqrt(mean_squared_error(test_data_aligned, test_predict.flatten()))
lstm_mae = mean_absolute_error(test_data_aligned, test_predict.flatten())
lstm_mape = np.mean(np.abs((test_data_aligned - test_predict.flatten()) / test_data_aligned)) * 100

# Get actual live price
live_data = yf.download(inp, period="1d", interval="1m")
actual_price = live_data['Close'].iloc[-1]

# Compare ARIMA and LSTM predictions
if abs(arima_predictions[-1] - actual_price) < abs(lstm_pred[0, 0] - actual_price):
    best_model = "ARIMA"
    best_pred = arima_predictions[-1]
    best_rmse = arima_rmse
    best_mae = arima_mae
    best_mape = arima_mape
else:
    best_model = "LSTM"
    best_pred = lstm_pred[0, 0]
    best_rmse = lstm_rmse
    best_mae = lstm_mae
    best_mape = lstm_mape

# Display results
st.subheader(f"Best Model: {best_model}")
st.write(f"Predicted share price for {inp} today by {best_model} is ${best_pred:.2f}")
st.write(f"Actual live price for {inp} is ${actual_price:.2f}")


# Plotting the closing price trend
st.subheader(f"Closing Price Trend for {inp}")
fig1, ax = plt.subplots(figsize=(12, 6))
ax.plot(data['Date'], data['Close'], label='Close', color='cyan')
ax.set_xlabel("Date")
ax.set_ylabel("Close Price")
ax.set_title("Closing Price", color='white')
ax.legend()
st.pyplot(fig1)

# Display predicted trend
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

# Create a third plot that zooms in on the last 30 days
zoom_start_date = data['Date'].max() - timedelta(days=30)
zoom_data = data[data['Date'] >= zoom_start_date]

st.subheader(f"Zoomed-in Predicted Closing Price for {inp}")
fig3, ax = plt.subplots(figsize=(12, 6))
ax.plot(zoom_data['Date'], zoom_data['Close'], label='Actual Price', color='cyan')
ax.plot(zoom_data['Date'], zoom_data['Test_Predict'], label='Predicted Price', color='red', linestyle='--')
ax.set_xlabel("Date")
ax.set_ylabel("Price")
ax.set_title("Zoomed-in Actual vs Predicted Closing Prices", color='white')
ax.legend()
