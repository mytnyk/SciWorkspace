import yfinance as yf

share = yf.Ticker("GME")

# get all stock info
share.info

# get historical market data
hist = share.history(period="max")
hist.to_csv("gme_stock_history.csv")

#print(hist)
# show meta information about the history (requires history() to be called first)
#print(share.history_metadata)