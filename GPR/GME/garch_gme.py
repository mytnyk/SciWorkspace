import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from arch import arch_model
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error

# Step 1: Load the data
df = pd.read_csv("_data_/gme_stock_history.csv")
df['Date'] = pd.to_datetime(df['Date'], utc=True)

#start_date = pd.to_datetime("2006-01-01", utc=True)
#end_date = pd.to_datetime("2006-02-01", utc=True)
start_date = pd.to_datetime("2020-10-01", utc=True)
end_date = pd.to_datetime("2022-02-01", utc=True)

df = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)]
X = df['Date']

df.set_index('Date', inplace=True)

# Using the closing price
horizon_period = 70
prices = df['Open']

n = len(prices)-horizon_period
# take test data:
test_prices = prices[n:]
# take training data:
prices = prices[:n]



# Step 2: Calculate daily returns
returns = prices.pct_change().dropna()

# Step 3: Fit an ARIMA model to the returns (optional)
arima_order = (1, 0, 1)  # Example order
arima_model = ARIMA(returns, order=arima_order)
arima_fit = arima_model.fit()


# Forecast future returns from ARIMA model
arima_forecast = arima_fit.forecast(steps=horizon_period)  # Forecasting horizon_period days of returns

# Step 4: Fit GARCH model to ARIMA residuals
garch_model = arch_model(arima_fit.resid, vol='Garch', p=1, q=1)
garch_fit = garch_model.fit()

# Forecast future volatility from GARCH model
garch_forecast = garch_fit.forecast(horizon=70)
predicted_volatility = np.sqrt(garch_forecast.variance.values[-1, :])

# Step 5: Combine ARIMA returns and GARCH volatility to forecast future prices
last_price = prices.iloc[-1]
predicted_returns = arima_forecast.values
forecasted_prices = [last_price * (1 + predicted_returns[i]) for i in range(horizon_period)]

# Compute confidence intervals
confidence_level = 1.96  # For a 95% confidence interval
upper_bounds = [forecasted_prices[i] * (1 + confidence_level * predicted_volatility[i]) for i in range(horizon_period)]
lower_bounds = [forecasted_prices[i] * (1 - confidence_level * predicted_volatility[i]) for i in range(horizon_period)]

# Step 6: Plot the results with error bars
plt.figure(figsize=(10, 6))

future_dates = pd.date_range(prices.index[-1], periods=horizon_period, freq='B')
plt.plot(future_dates, forecasted_prices, label="Прогноз", color='blue')
#plt.plot(future_dates, test_prices, label="Тест", color='green')
#plt.plot(prices.index, prices, label="Тренування", color="black")

plt.scatter(prices.index, prices, label="Тренування", marker=".", c="k")
plt.scatter(future_dates, test_prices, label="Тест", marker="+", c="g")


plt.fill_between(future_dates, lower_bounds, upper_bounds, alpha=0.2, label="95% інтервал довіри")#color="lightblue"
#plt.xlabel("Date")
plt.ylabel("Ціна, $", rotation=0, labelpad=15)
#plt.title("Forecasted Share Prices with Confidence Interval")
plt.legend()
plt.show()

rmse = np.sqrt(mean_squared_error(test_prices, forecasted_prices))
print("RMSE:", rmse)
