import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import date, timedelta, datetime
import streamlit as st
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import roc_auc_score, roc_curve
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

st.title("StockViews")
st.subheader("Stock Trend Prediction with Hybrid Model (LSTM + ARIMA)")
# Asks the user to input stock ticker
inp = st.text_input('Enter Stock Ticker', 'AAPL')  

if not inp.strip():
    st.error("Error: The stock ticker cannot be blank. Please enter a valid stock ticker.")
else:
    end = date.today()
    
    try:
        # Fetching all data using yfinance 
        data_all = yf.download(inp, end=end)
        if data_all.empty:
            st.error("Error: The stock ticker is incorrect or data is not available for the given ticker.")
        else:
            data_all.reset_index(inplace=True)
            data_all.rename(columns={'Adj Close': 'Adj_Close'}, inplace=True) 

            # Determine the start date based on the data length
            total_years = (data_all['Date'].iloc[-1] - data_all['Date'].iloc[0]).days / 365
            if total_years > 5:
                start = end - timedelta(days=4*365)
            else:
                start = end - timedelta(days=2*365)

            # Filter the data for the specified date range
            data = data_all[data_all['Date'] >= pd.to_datetime(start)]

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
                    model = ARIMA(history, order=(2, 1, 1))
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

            # Get actual live price
            live_data = yf.download(inp, period="1d", interval="1m")
            actual_price = live_data['Close'].iloc[-1]

            # Compare ARIMA and LSTM predictions
            if abs(arima_predictions[-1] - actual_price) < abs(lstm_pred[0, 0] - actual_price):
                best_model = "ARIMA"
                best_pred = arima_predictions[-1]
                data['Best_Predict'] = np.nan
                data.iloc[-len(arima_predictions):, data.columns.get_loc('Best_Predict')] = arima_predictions
            else:
                best_model = "LSTM"
                best_pred = lstm_pred[0, 0]
                data['Best_Predict'] = data['Test_Predict']

            # Determine buy/sell recommendation
            recommendation = "Our prediction model shows that the predicted price of the stock is equal to the current actual price. This indicates that there is no expected change in the stock's value in the near future. In this case, we recommend you hold onto your stock, as there is no anticipated movement that would warrant buying or selling at this time. However, please note that prediction models aren't always accurate due to sudden market changes, so this is only a recommendation."
            if best_pred > actual_price:
                recommendation = "Our prediction model indicates that the predicted price of the stock is higher than the current actual price. This suggests a potential increase in the stock's value. Based on this information, we recommend you consider buying the stock, as it is expected to appreciate, potentially offering you a profitable investment opportunity. However, please note that prediction models aren't always accurate due to sudden market changes, so this is only a recommendation."
            elif best_pred < actual_price:
                recommendation = "According to our prediction model, the predicted price of the stock is lower than the current actual price. This implies a potential decrease in the stock's value. Therefore, we advise you to consider selling the stock to avoid potential losses. Selling now could help you preserve your capital and avoid a decrease in your investment's value. However, please note that prediction models aren't always accurate due to sudden market changes, so this is only a recommendation."

            # Calculate binary classification metrics (up or down)
            data['Actual_Direction'] = data['Close'].diff().apply(lambda x: 1 if x > 0 else 0)
            data['Predicted_Direction'] = data['Best_Predict'].diff().apply(lambda x: 1 if x > 0 else 0)

            # Ensure alignment for classification metrics
            valid_indices = ~data['Predicted_Direction'].isna() & ~data['Actual_Direction'].isna()
            actual_direction = data.loc[valid_indices, 'Actual_Direction']
            predicted_direction = data.loc[valid_indices, 'Predicted_Direction']

            auc_roc = roc_auc_score(actual_direction, predicted_direction)

            # Display results
            st.subheader(f"Best Model: {best_model}")
            st.write(f"Predicted share price for {inp} today by {best_model} is ${best_pred:.2f}")
            st.write(f"Actual live price for {inp} is ${actual_price:.2f}")
            

            # Plotting the closing trend
            st.subheader(f"Closing Trend for {inp}")
            fig1, ax1 = plt.subplots()
            ax1.plot(data['Date'], data['Close'], label='Close Price')
            ax1.set_xlabel('Date')
            ax1.set_ylabel('Price')
            ax1.legend()
            st.pyplot(fig1)

            # Plotting the predicted trend based on the best model
            st.subheader(f"Predicted Trend for {inp}")
            fig2, ax2 = plt.subplots()
            ax2.plot(data['Date'], data['Close'], label='Actual Close Price')
            ax2.set_xlabel('Date')
            ax2.set_ylabel('Price')
            ax2.legend()
            st.pyplot(fig2)

            # Plotting the zoomed-in plot of the last 30 days
            st.subheader(f"Last 30 Days Prediction vs Actual for {inp}")
            fig3, ax3 = plt.subplots()
            ax3.plot(data['Date'].iloc[-30:], data['Close'].iloc[-30:], label='Actual Close Price')
            ax3.set_xlabel('Date')
            ax3.set_ylabel('Price')
            ax3.legend()
            st.pyplot(fig3)

            st.write(f"Recommendation for {inp}: {recommendation}")

    except Exception as e:
        st.error(f"An error occurred: {e}")
