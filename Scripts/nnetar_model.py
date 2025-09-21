import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from scipy.stats import boxcox
from scipy.special import inv_boxcox
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout

def create_sequences(data, seq_length):
    """Creates sequences for neural network input."""
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(X), np.array(y)

def run_nnetar_model(df):
    """Fits a neural network model and forecasts future prices."""
    print("--- Running NNETAR-like Model ---")
    close_prices = df['Close'].values.reshape(-1, 1)

    # Box-Cox Transformation
    close_prices_bc, lambda_val = boxcox(close_prices.flatten())
    close_prices_bc = close_prices_bc.reshape(-1, 1)

    # Normalization
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(close_prices_bc)

    # Create sequences
    seq_length = 11
    X_data, y_data = create_sequences(scaled_data, seq_length)
    
    # Reshape for LSTM: [samples, time_steps, features]
    X_data = X_data.reshape((X_data.shape[0], X_data.shape[1], 1))

    # Build and compile the model
    model = Sequential([
        LSTM(6, activation='relu', input_shape=(seq_length, 1)),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_data, y_data, epochs=100, batch_size=32, verbose=0)

    # Forecast for the next 30 days
    future_predictions = []
    last_sequence = scaled_data[-seq_length:]
    for _ in range(30):
        current_pred = model.predict(last_sequence.reshape(1, seq_length, 1), verbose=0)
        future_predictions.append(current_pred[0][0])
        last_sequence = np.append(last_sequence[1:], current_pred).reshape(seq_length, 1)

    # Inverse transform
    future_predictions_scaled = np.array(future_predictions).reshape(-1, 1)
    future_predictions_inv = scaler.inverse_transform(future_predictions_scaled)
    future_predictions_final = inv_boxcox(future_predictions_inv.flatten(), lambda_val)

    # Plotting
    future_dates = pd.date_range(start=df.index[-1], periods=31)[1:]
    plt.figure(figsize=(10, 6))
    plt.plot(df.index, df['Close'], label='Historical Data')
    plt.plot(future_dates, future_predictions_final, color='red', linestyle='--', label='Neural Net Forecast')
    plt.title('Neural Net 30-Day Forecast')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    df = yf.download('^GSPC', start='2015-01-01', end='2020-06-04')
    run_nnetar_model(df)