import pandas as pd

df = pd.read_csv('data/product_view_data.csv')
print(df.head())

print(df.info())


###################
# Missing data

# Replace missing values with mean
mean = df['view_duration'].mean()
df['view_duration'] = df['view_duration'].fillna(mean)
# Alternative to use the same command in place:
# df['view_duration'].fillna(mean, inplace = True)


###################
# Duplicates

# Find and drp duplicates
print(df.duplicated())
print(sum(df.duplicated()))
df.drop_duplicates(inplace=True)
print(df.head())


###################
# Wrong data types

# Incorrect data types
df['timestamp'] = pd.to_datetime(df['timestamp'])
print(df.info())
