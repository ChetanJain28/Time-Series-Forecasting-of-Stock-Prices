# main.py
import yfinance as yf
import data_visualization as dv
import arima_model as am
import garch_model as gm
import prophet_model as pm
import knn_model as km
import nnetar_model as nm

def run_all_analysis():
    """Central function to run all time series forecasting analyses."""
    
    # 1. Download and visualize the data
    print("--- 📥 Fetching S&P 500 data... ---")
    df = yf.download('^GSPC', start='2015-01-01', end='2020-06-04')
    dv.run_data_visualization()

    # 2. Run the individual models
    print("\n--- 🧠 Running Time Series Models ---")
    am.run_arima_model(df)
    gm.run_garch_model(df)
    pm.run_prophet_model(df)
    km.run_knn_model(df)
    nm.run_nnetar_model(df)
    
    print("\n--- ✅ All analyses complete. ---")

if __name__ == "__main__":
    run_all_analysis()