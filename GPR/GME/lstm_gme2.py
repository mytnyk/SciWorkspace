
from tensorflow.keras.layers import Input, SimpleRNN, GRU, LSTM, Dense, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD, Adam

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

start_date = pd.to_datetime("2020-10-01", utc=True)
end_date = pd.to_datetime("2022-02-01", utc=True)

data_history = pd.read_csv('_data_/gme_stock_history.csv')#.to_numpy()
data_history['Date'] = pd.to_datetime(data_history['Date'], utc=True)  # Convert 'Date' to datetime
data_history = data_history[(data_history['Date'] >= start_date) & (data_history['Date'] <= end_date)]

# Using the closing price
series = data_history['Close'].to_numpy()

# make the original data
#series = np.sin((0.1*np.arange(400))**2)


# plot it
plt.plot(series)
plt.show()


### build the dataset
# let's see if we can use T past values to predict the next value
T = 30
D = 1
X = []
Y = []
for t in range(len(series) - T):
  x = series[t:t+T]
  X.append(x)
  y = series[t+T]
  Y.append(y)

X = np.array(X).reshape(-1, T) # make it N x T
Y = np.array(Y)
N = len(X)
print("X.shape", X.shape, "Y.shape", Y.shape)

### Now try RNN/LSTM model
X = X.reshape(-1, T, 1) # make it N x T x D

# make the RNN
i = Input(shape=(T, D))
x = LSTM(10)(i)
x = Dense(1)(x)
model = Model(i, x)
model.compile(
  loss='mse',
  optimizer=Adam(learning_rate=0.05),
)

split = 70#N//2
#split = len(X)-
#X_train, X_test = X[:split], X[split:]
#y_train, y_test = y[:split], y[split:]

# train the RNN
r = model.fit(
  X[:-split], Y[:-split],
  batch_size=32,
  epochs=200,
  validation_data=(X[-split:], Y[-split:]),
)


# Multi-step forecast
forecast = []
input_ = X[-split]
while len(forecast) < len(Y[-split:]):
  # Reshape the input_ to N x T x D
  f = model.predict(input_.reshape(1, T, 1))[0,0]
  forecast.append(f)

  # make a new input with the latest forecast
  input_ = np.roll(input_, -1)
  input_[-1] = f

#plt.plot(Y[-split:], label='targets')
plt.plot(range(len(series)), series, label='targets')
plt.plot(range(len(series) - split, len(series)), forecast, label='forecast')
plt.title("RNN Forecast")
plt.legend()
plt.show()