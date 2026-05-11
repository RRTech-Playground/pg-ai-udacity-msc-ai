# imports and load data
import pandas as pd
from matplotlib import pyplot as plt

df = pd.read_csv('data/store_data.csv')
print(df.head())

#######################
# Draw Conclusions Quiz

# explore data by creating histograms on the entire DataFrame
df.hist(figsize=(8, 8))
#plt.show()

# Use tail to find the end of the dataset to locate where the last month is via its index
print(df.tail(20))

# Use iloc to create a slice of the last month and sum up the weeks to find the total sales for the last month
print(df.iloc[196:, 1:].sum())

# Use the mean method to find the average sales for each store
print(df.mean(numeric_only=True))

# Find the sales of all stores by filtering on the week of march 13, 2016
print(df[df['week'] == '2016-03-13'])

# Use the min method to filter the dataset to find the worst week for store C
print(df[df['storeC'] == df['storeC'].min()])

# Filter the DataFrame on the most recent 3 month period. You can filter by selecting greater than or equal to 2017-12-01
last_three_months = df[df['week'] >= '2017-12-01']

# Find the total sales during this 3 month
print(last_three_months.iloc[:, 1:].sum())  # exclude sum of week column

