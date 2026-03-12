import pandas as pd

df = pd.read_csv('data/census_income_data.csv')
print(df.head())

# Try finding the overall size of the dataframe. There is a method
# that will output something like (42, 5)
print(df.shape)

# Which method can you use to find all of the column types as well
# as how many non-null values there are for each?
print(df.info())
#print(df.info)  # shows the columns, but we are looking for the rows

# To find out the actual type of an obeject
print(type(df['workclass'][0]))
print(type(df['education'][0]))
print(type(df['relationship'][0]))
print(type(df['income'][0]))

# Use a dataframe method to find the unique values for the column education.
print(df['education'].nunique())

# What dataframe method outputs all of the summary statistics for
# each column, including mean, min, and max?
print(df.describe())
