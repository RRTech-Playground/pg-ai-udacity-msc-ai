# import pandas and read the two data csvs
import pandas as pd

fuel_08_df = pd.read_csv('data/data_08_v3.csv')
fuel_18_df = pd.read_csv('data/data_18_v3.csv')

# Find the data types
print(fuel_08_df.dtypes)
print(fuel_18_df.dtypes)


#######################
# Inspecting Data Types


# What is different between the two datasets for the cyl column? What would you change?

# Find the unique values in cyl
print(fuel_08_df.cyl.unique())
print(fuel_18_df.cyl.unique())


# What is different between the two datasets for the air_pollution_score column? What would you change?

# Find the unique values in air_pollution_score
print(fuel_08_df.air_pollution_score.unique())
print(fuel_18_df.air_pollution_score.unique())


# Look at the data types for all of th *_mps columns for both datasets. What type are they? What would you change?

# Find the unique values in city_mpg
print(fuel_08_df.city_mpg.unique())
print(fuel_18_df.city_mpg.unique())


# What is different between the two datasets for the`greenhouse_gas_score`column? What would you change?

# Find the unique values in greenhouse_gas_score
print(fuel_08_df.greenhouse_gas_score.unique())
print(fuel_18_df.greenhouse_gas_score.unique())


#######################
# Fixing Data Types

df_08 = fuel_08_df
df_18 = fuel_18_df

# check value counts for the 2008 cyl column
print(df_08['cyl'].value_counts())


## String processing with regex
# We need to extract the number from the string.

# Extract int from strings in the 2008 cyl column
df_08['cyl'] = df_08['cyl'].str.extract('(\d+)').astype(int)

# Check value counts for 2008 cyl column again to confirm the change
print(df_08['cyl'].value_counts())

# convert 2018 cyl column from a float type to int type
df_18['cyl'] = df_18['cyl'].astype(int)


## Fixing `air_pollution_score` Data Type
# - 2008: convert string to float
# - 2018: convert int to float

# try using pandas' astype function to convert the
# 2008 air_pollution_score column to float -- this won't work
# df_08.air_pollution_score = df_08.air_pollution_score.astype(float)

# ValueError: could not convert string to float: '6/4'

print(df_08[df_08.air_pollution_score == '6/4'])

# It's not just the air pollution score - some columns hold two values for two (for two fuel types)

# First, let's get all the hybrids in 2008
hb_08 = df_08[df_08['fuel'].str.contains('/')]
print(hb_08.shape)

# hybrids in 2018
hb_18 = df_18[df_18['fuel'].str.contains('/')]
print(hb_18.shape)

# create two copies of the 2008 hybrids dataframe
df1 = hb_08.copy()  # data on first fuel type of each hybrid vehicle
df2 = hb_08.copy()  # data on second fuel type of each hybrid vehicle

# Each one should look like this
print(df1)

# columns to split by "/"
split_columns = ['fuel', 'air_pollution_score', 'city_mpg', 'hwy_mpg', 'cmb_mpg', 'greenhouse_gas_score']

# apply split function to each column of each dataframe copy
for c in split_columns:
    df1[c] = df1[c].apply(lambda x: x.split("/")[0])
    df2[c] = df2[c].apply(lambda x: x.split("/")[1])

print(df1)
print(df2)

# combine dataframes to add to the original dataframe
new_rows = pd.concat([df1, df2])

# now we have separate rows for each fuel type of each vehicle!
print("\n")
print(new_rows)

# drop the original hybrid rows
df_08.drop(hb_08.index, inplace=True)

# add in our newly separated rows
df_08 = pd.concat([df_08, new_rows], ignore_index=True)

# check that all the original hybrid rows with "/"s are gone
print(df_08[df_08['fuel'].str.contains('/')])

print(df_08.shape)

# 2018
# create two copies of the 2018 hybrids dataframe, hb_18
df1 = hb_18.copy()
df2 = hb_18.copy()

# ### Split values for fuel, city_mpg, hwy_mpg, cmb_mpg []
# You don't need to spli for air_pollution_score or greenhouse_gas_score here because these columns are already ints in the 2018 dataset.

# list of columns to split
split_columns = ['fuel', 'city_mpg', 'hwy_mpg', 'cmb_mpg']

# apply split function to each column of each dataframe copy
for c in split_columns:
    df1[c] = df1[c].apply(lambda x: x.split("/")[0])
    df2[c] = df2[c].apply(lambda x: x.split("/")[1])

# append the two dataframes
new_rows = pd.concat([df1, df2])

# drop each hybrid row from the original 2018 dataframe
# do this by using Pandas drop function with hb_18's index
df_18.drop(hb_18.index, inplace=True)

# append new_rows to df_18
df_18 = pd.concat([df_18, new_rows], ignore_index=True)

# check that they're gone
print(df_18[df_18['fuel'].str.contains('/')])
print(df_18.shape)

# convert string to float for 2008 air pollution column
df_08.air_pollution_score = df_08.air_pollution_score.astype(float)

# convert int to float for 2018 air pollution column
df_18.air_pollution_score = df_18.air_pollution_score.astype(float)

# convert string to float for 2008 air pollution column
df_08.air_pollution_score = df_08.air_pollution_score.astype(float)

# convert int to float for 2018 air pollution column
df_18.air_pollution_score = df_18.air_pollution_score.astype(float)

# convert mpg columns to floats
mpg_columns = ['city_mpg', 'hwy_mpg', 'cmb_mpg']
for c in mpg_columns:
    df_18[c] = df_18[c].astype(float)
    df_08[c] = df_08[c].astype(float)

# Fixgreenhouse_gas_score datatype - convert from float to int
df_08['greenhouse_gas_score'] = df_08['greenhouse_gas_score'].astype(int)

# Take one last check to confirm all the changes.
print(df_08.dtypes)
print(df_18.dtypes)
print(df_08.dtypes == df_18.dtypes)

# Save your final CLEAN datasets as new files!
df_08.to_csv('data/clean_08.csv', index=False)
df_18.to_csv('data/clean_18.csv', index=False)


