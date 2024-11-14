# THIS IS EXAMPLE OF BAD PREDICTION
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import StandardScaler

# Generate synthetic data
#time = np.arange(0, 100, 0.1)
#data = np.sin(time) + np.random.normal(0, 0.1, len(time))

start_date = pd.to_datetime("2020-10-01", utc=True)
end_date = pd.to_datetime("2022-02-01", utc=True)


data_history = pd.read_csv('_data_/gme_stock_history.csv')#.to_numpy()
data_history['Date'] = pd.to_datetime(data_history['Date'], utc=True)  # Convert 'Date' to datetime
data_history = data_history[(data_history['Date'] >= start_date) & (data_history['Date'] <= end_date)]
#X_ordinal = data_history['Date'].map(pd.Timestamp.toordinal).values.reshape(-1, 1)


# Using the closing price
data = data_history['Close'].to_numpy()

# Prepare the dataset
df = pd.DataFrame(data, columns=['value'])
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df)

#null last 100 data
scaled_data[-100:] = 0

# Create the sequences
def create_sequences(data, seq_length):
    sequences = []
    targets = []
    for i in range(len(data) - seq_length):
        sequences.append(data[i:i + seq_length])
        targets.append(data[i + seq_length])
    return np.array(sequences), np.array(targets)

seq_length = 3
X, y = create_sequences(scaled_data, seq_length)
X = X.reshape((X.shape[0], X.shape[1], 1))

# Split into train and test sets
split = len(X)-100
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# Build the LSTM model
model = Sequential()
model.add(LSTM(50, activation='relu', input_shape=(seq_length, 1)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')

# Train the model
history = model.fit(X_train, y_train, epochs=50, verbose=1)#validation_data=(X_test, y_test),

# Predict and plot
y_pred = model.predict(X_test)
y_pred_inv = scaler.inverse_transform(y_pred)
y_test_inv = scaler.inverse_transform(y_test)

y_train_inv = scaler.inverse_transform(y_train)
t_train = np.arange(len(y_train))
t_test = np.arange(len(y_train), len(y_train) + len(y_test))
plt.plot(t_train, y_train_inv, label='Train')

plt.plot(t_test, y_test_inv, label='True Value')
plt.plot(t_test, y_pred_inv, label='Predicted Value')
plt.legend()
plt.show()

#input();