import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('./GOOG.csv')

print(df.head())

google_stock = pd.read_csv('GOOG.csv', index_col = ['Date'],  parse_dates = True, usecols = ['Date', 'Adj Close'])
apple_stock = pd.read_csv('AAPL.csv',  index_col = ['Date'],  parse_dates = True, usecols = ['Date', 'Adj Close'])
amazon_stock = pd.read_csv('AMZN.csv', index_col = ['Date'],  parse_dates = True, usecols = ['Date', 'Adj Close'])

print(google_stock.head(2))

dates = pd.date_range('2000-01-01', '2016-12-31')
all_stocks = pd.DataFrame(index = dates)

google_stock = google_stock.rename(columns = {'Adj Close' : 'Google'})
apple_stock = apple_stock.rename(columns = {'Adj Close' : 'Apple'})
amazon_stock = amazon_stock.rename(columns = {'Adj Close' : 'Amazon'})

print(google_stock.head(2))

all_stocks = all_stocks.join(google_stock)
all_stocks = all_stocks.join(apple_stock)
all_stocks = all_stocks.join(amazon_stock)

print(all_stocks.head())
print(all_stocks.describe())

print(all_stocks.isnull().sum())
all_stocks.dropna(axis=0, inplace=True)
print(all_stocks.isnull().sum())

print('\n')
print('The average stock price for each stock is: \n', all_stocks.mean(), '\n')
# Note: You can get the same result by printing it individually one-by-one as `all_stocks['Google'].mean()`
print('The median stock price for each stock is: \n', all_stocks.median(), '\n')
print('The standard deviation of the stock price for each stock  is: \n', all_stocks.std(), '\n')
print('The correlation between stocks is: \n', all_stocks.corr(), '\n')

print('\n')
rollingMean = all_stocks['Google'].rolling(150).mean()
print(rollingMean)

# We plot the Google stock data
plt.plot(all_stocks['Google'])

# We plot the rolling mean ontop of our Google stock data
plt.plot(rollingMean)
plt.legend(['Google Stock Price', 'Rolling Mean'])
plt.show()
