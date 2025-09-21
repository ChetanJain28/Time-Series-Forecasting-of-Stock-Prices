import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from arch import arch_model

def run_garch_model(df):
    """Fits a GARCH model and forecasts volatility."""
    print("--- Running GARCH Model ---")
    close_prices = df['Close']
    returns = 100 * close_prices.pct_change().dropna()

    # Fitting GARCH(1,1) with ARMA mean model
    garch_model = arch_model(returns, mean='AR', lags=3, vol='Garch', p=1, q=1)
    garch_fit = garch_model.fit(disp='off')
    print(garch_fit.summary())

    # Conditional Volatility Plot
    plt.figure(figsize=(10, 6))
    plt.plot(garch_fit.conditional_volatility)
    plt.title('Conditional Volatility Plot')
    plt.ylabel('Sigma(t)')
    plt.show()

    # Forecasting returns and volatility
    forecasts = garch_fit.forecast(horizon=30, reindex=False)
    forecast_volatility = np.sqrt(forecasts.variance.values[-1, :])
    forecast_mean = forecasts.mean.values[-1, :]

    plt.figure(figsize=(10, 6))
    plt.plot(returns.index, returns, label='Returns')
    plt.plot(pd.date_range(start=returns.index[-1], periods=30, freq='B'),
             forecast_mean, color='red', label='Forecast Mean')
    plt.fill_between(pd.date_range(start=returns.index[-1], periods=30, freq='B'),
                     -forecast_volatility, forecast_volatility, color='gray', alpha=0.5, label='Volatility')
    plt.title('GARCH 30-Day Forecast of Returns')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    df = yf.download('^GSPC', start='2015-01-01', end='2020-06-04')
    run_garch_model(df)