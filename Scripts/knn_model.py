import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error

def create_lagged_features(data, lags):
    """Creates lagged features for K-NN model."""
    X, y = [], []
    for i in range(len(data) - lags):
        X.append(data[i:i + lags])
        y.append(data[i + lags])
    return np.array(X), np.array(y)

def run_knn_model(df):
    """Fits a K-NN model and forecasts future prices."""
    print("--- Running K-NN Model ---")
    
    # Fix: Ensure the data is a 1D array before passing it to the function
    close_prices = df['Close'].values.flatten()
    lags = 30
    X, y = create_lagged_features(close_prices, lags)
    
    # Fix: Reshape the data to a 2D array
    X = X.reshape(-1, lags)
    y = y.reshape(-1, 1)

    # Train on the full dataset for forecasting
    knn_model = KNeighborsRegressor(n_neighbors=50)
    knn_model.fit(X, y)

    # Multi-step forecasting
    future_predictions = []
    current_input = X[-1].reshape(1, -1)
    for _ in range(30):
        next_pred = knn_model.predict(current_input)[0][0]
        future_predictions.append(next_pred)
        current_input = np.append(current_input[:, 1:], next_pred).reshape(1, -1)

    future_dates = pd.date_range(start=df.index[-1], periods=31)[1:]
    plt.figure(figsize=(10, 6))
    plt.plot(df.index, df['Close'], label='Historical Data')
    plt.plot(future_dates, future_predictions, color='red', linestyle='--', label='K-NN Forecast')
    plt.title('K-NN 30-Day Forecast')
    plt.legend()
    plt.show()

    # Rolling Origin Accuracy Evaluation
    tscv = TimeSeriesSplit(n_splits=5)
    mse_scores = []
    
    for train_index, test_index in tscv.split(X):
        X_train_cv, X_test_cv = X[train_index], X[test_index]
        y_train_cv, y_test_cv = y[train_index], y[test_index]
        knn_model.fit(X_train_cv, y_train_cv)
        y_pred_cv = knn_model.predict(X_test_cv)
        mse_scores.append(mean_squared_error(y_test_cv, y_pred_cv))
    
    print(f"Rolling Origin RMSE: {np.sqrt(np.mean(mse_scores))}")

if __name__ == '__main__':
    df = yf.download('^GSPC', start='2015-01-01', end='2020-06-04')
    run_knn_model(df)