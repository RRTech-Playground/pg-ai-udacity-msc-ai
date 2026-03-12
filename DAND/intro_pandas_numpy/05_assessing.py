import pandas as pd

df = pd.read_csv('data/cancer_data.csv')
print(df.head())

# this returns a tuple of the dimensions of the dataframe
print(df.shape)

# this returns the datatypes of the columns
print(df.dtypes)

# although the datatype for diagnosis appears to be object,
# further investigation shows it's a string
print(type(df['diagnosis'][0]))

# this displays a concise summary of the dataframe,
# including the number of non-null values in each column
print(df.info())

# this returns the number of unique values in each column
print(df.nunique())

# this returns useful descriptive statistics for each column of data
print(df.describe())

# this returns the first few lines in our dataframe
# by default, it returns the first five
print(df.head())

# although, you can specify however many rows you'd like returned
print(df.head(20))

# same thing applies to `.tail()` which returns the last few rows
print(df.tail(2))

## Indexing and Selecting Data in Pandas¶

# Let's separate this dataframe into three new dataframes - one for each metric (mean, standard error, and maximum).
# To get the data for each dataframe, we need to select the id and diagnosis columns, as well as the ten columns for that metric.
for i, v in enumerate(df.columns):
    print(i, v)

# We can select data using loc and iloc, which you can read more about here. loc uses labels of rows or columns to select data,
# while iloc uses the index numbers. We'll use these to index the dataframe below.

# select all the columns from 'id' to the last mean column
df_means = df.loc[:,'id':'fractal_dimension_mean']
print(df_means.head())

# repeat the step above using index numbers
df_means = df.iloc[:,:12]
print(df_means.head())

# Let's save the dataframe of means for use in a future notebook.
df_means.to_csv('data/cancer_data_means.csv', index=False)

## Selecting Multiple Ranges in Pandas

# Selecting the columns for the mean dataframe was pretty straightforward - the columns we needed to select were all together (id, diagnosis, and the mean columns). Now we run into a little issue when we try to do the same for the standard errors or maximum values. id and diagnosis are separated from the rest of the columns we need! We can't specify all of these in one range.
# First, try creating the standard error dataframe on your own to see why doing this with just loc and iloc is an issue. While that is cumbersome, there is another solution.
# One solution is to use NumPy's .r_[] method which translates slice objects to concatenation along the first axis. You can read more here.
# >>> np.r_[:2, 4:6]
# array([0, 1, 4, 5])
# How did I find this solution? I found this stackoverflow link by googling "how to select multiple ranges df.iloc -> https://stackoverflow.com/questions/41256648/select-multiple-ranges-of-columns-in-pandas-dataframe".
# Once we have the index of the columns we want, we can use .iloc to select our multiple ranges.

# import
import numpy as np

# create the standard errors dataframe
df_SE = df.iloc[:, np.r_[:2, 12:22]]

# view the first few rows to confirm this was successful
print('--')
print(df_SE.head())
