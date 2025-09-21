import pandas as pd
import yfinance as yf
from prophet import Prophet
import matplotlib.pyplot as plt

def run_prophet_model(df):
    """Fits a Prophet model and forecasts future prices."""
    print("--- Running Prophet Model ---")
    
    # The fix is here: explicitly select the 'Close' column as a Series
    # The 'Close' column might be a 2D array, so convert it to a 1D Series
    df_prophet = pd.DataFrame({'ds': df.index, 'y': df['Close'].values.flatten()})

    model = Prophet()
    model.fit(df_prophet)

    future = model.make_future_dataframe(periods=30)
    forecast = model.predict(future)

    fig1 = model.plot(forecast)
    plt.title('Prophet Forecast')
    plt.show()

    fig2 = model.plot_components(forecast)
    plt.show()

if __name__ == '__main__':
    # For standalone execution, fetch data
    df = yf.download('^GSPC', start='2015-01-01', end='2020-06-04', auto_adjust=False)
    df.columns = [col.replace('GSPC.', '') for col in df.columns]
    run_prophet_model(df)