import pandas as pd
import matplotlib.pyplot as plt

data_history = pd.read_csv('_data_/gme_stock_history.csv')#.to_numpy()

data_history['Date'] = pd.to_datetime(data_history['Date'], utc=True)  # Convert 'Date' to datetime

#start_date = pd.to_datetime("2020-12-01", utc=True)
#end_date = pd.to_datetime("2021-03-01", utc=True)
#data_history = data_history[(data_history['Date'] >= start_date) & (data_history['Date'] <= end_date)]

X = data_history['Date']
Y = data_history['Open'].values.reshape(-1, 1) 

y = Y.flatten()
plt.plot(X, y, label=r"GME")#, linestyle="dotted"
#plt.legend()
#plt.xlabel("Час")
plt.ylabel("Ціна, $", rotation=0, labelpad=15)
#_ = plt.title("GameStop Corp. (GME)")
plt.show()
