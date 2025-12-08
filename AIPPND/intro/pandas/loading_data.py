import pandas as pd

google_stock = pd.read_csv('./GOOG.csv')
print(type(google_stock))
print(google_stock.shape)

print(google_stock)

print(google_stock.head())
print(google_stock.tail())

print(google_stock.isnull().any())
print(google_stock.describe())
print(google_stock['Adj Close'].describe())

print()
print(google_stock.max())
print(google_stock['Close'].min())
print(google_stock.mean(numeric_only=True))

print()
print()
print(google_stock.corr(numeric_only=True))

data = pd.read_csv('./fake_company.csv')
print(data)

# Display the total amount of money spent in salaries each year
print(data.groupby(['Year'])['Salary'].sum())

# We want to know what was the average salary for each year.
print(data.groupby(['Year'])['Salary'].mean())

# Let's see how much did each employee gets paid in those three years
print(data.groupby(['Name'])['Salary'].sum())

# Let's see what was the salary distribution per department per year.
print(data.groupby(['Year', 'Department'])['Salary'].sum())
