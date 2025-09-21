import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

def run_data_visualization():
    """Performs data extraction, visualization, and stationarity tests."""
    print("--- Running Data Visualization ---")
    ticker = '^GSPC'
    # Download data with all columns to get Open, High, Low, Close, and Volume
    df = yf.download(ticker, start='2015-01-01', end='2020-06-04', auto_adjust=False)
    
    # Calculate simple moving averages for Bollinger Bands
    df['20d_ma'] = df['Close'].rolling(window=20).mean()
    df['20d_std'] = df['Close'].rolling(window=20).std()
    df['upper_band'] = df['20d_ma'] + (df['20d_std'] * 2)
    df['lower_band'] = df['20d_ma'] - (df['20d_std'] * 2)
    
    # Calculate MACD
    df['ema_12'] = df['Close'].ewm(span=12, adjust=False).mean()
    df['ema_26'] = df['Close'].ewm(span=26, adjust=False).mean()
    df['macd'] = df['ema_12'] - df['ema_26']
    df['signal_line'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_histogram'] = df['macd'] - df['signal_line']
    
    # Plotting
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(df.index, df['Close'], label='Close Price', color='black')
    ax.plot(df.index, df['upper_band'], label='Upper Bollinger Band', color='blue', linestyle='--')
    ax.plot(df.index, df['lower_band'], label='Lower Bollinger Band', color='blue', linestyle='--')
    ax.fill_between(df.index, df['lower_band'], df['upper_band'], alpha=0.1, color='blue')
    ax.set_title('S&P 500 Daily Close Price with Bollinger Bands')
    ax.legend()
    plt.show()

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True, gridspec_kw={'height_ratios': [3, 1]})
    ax1.plot(df.index, df['Close'], label='Close Price', color='black')
    ax2.plot(df.index, df['macd'], label='MACD', color='blue')
    ax2.plot(df.index, df['signal_line'], label='Signal Line', color='red')
    ax2.bar(df.index, df['macd_histogram'], label='MACD Histogram', color='gray')
    ax1.set_title('S&P 500 Daily Close Price')
    ax2.set_title('Moving Average Convergence Divergence (MACD)')
    ax1.legend()
    ax2.legend()
    plt.show()

    # ADF Test for stationarity
    close_prices = df['Close']
    adf_result = adfuller(close_prices)
    print(f"ADF Statistic: {adf_result[0]}")
    print(f"p-value: {adf_result[1]}")

    # Plot ACF and PACF
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    plot_acf(close_prices, ax=axes[0])
    plot_pacf(close_prices, ax=axes[1])
    plt.show()

    return df

if __name__ == '__main__':
    run_data_visualization()