# import pandas and read the two data csvs
import pandas as pd


# load 2008 and 2018 datasets
df_08 = pd.read_csv("data/clean_08.csv")
df_18 = pd.read_csv("data/clean_18.csv")

#######################
# Concatenation

# pd.concat([df1, df2], axis=0) # vertical
# pd.concat([df3, df4], axis=1) # horizontal

# Create two arrays as long as the number of rows in the 2008 and 2018 dataframes. Each array will repeat the value “2008” or “2018” corresponding with the DataFrame you are assigning.
# Note: If you had different values for each row in the new column, you'd have to assign the column an array the exact size as the DataFrame, but because we are only using one value, this is a nice little trick!
df_08['year'] = '2008'
print(df_08.head())

df_18['year'] = '2018'
print(df_18.head())

# concat dataframes
df = pd.concat([df_08, df_18])

# view dataframe to check for success
print(df.head(1))
#        model  displ  cyl  ... greenhouse_gas_score smartway  year
# 0  ACURA MDX    3.7    6  ...                    4       no  2008

print(df.tail(1))
#            model  displ  cyl ... greenhouse_gas_score smartway  year
# 831  VOLVO XC 90    2.0    4 ...                   10    Elite  2018

# safe concatenated data set
df.to_csv("data/clean_08_18_concatenated.csv", index=False)


#######################
# Join

# 1. Inner Join - Use intersection of keys from both frames.
# 2. Outer Join - Use union of keys from both frames.
# 3. Left Join - Use keys from left frame only.
# 4. Right Join - Use keys from right frame only.

# rename 2008 columns
df_08.rename(columns=lambda x: x[:10] + "_2008", inplace=True)

# view to check names
print(df_08.head())

# merge datasets
df_combined = df_08.merge(df_18, left_on='model_2008', right_on='model', how='inner')

# view to check merge
print(df_combined.head())

# safe merged data set
df_combined.to_csv('data/combined_dataset.csv', index=False)


#######################
# Results with a merged df

# Question: For all of the models that were produced in 2008 that are still being produced now, how much has the mpg improved and which vehicle improved the most?

df = df_combined

# find the mean `cmb_mpg_2008` and mean `cmb_mpg` for each
model_mpg = df.groupby('model').mean(numeric_only=True)[['cmb_mpg_2008', 'cmb_mpg']]

print(model_mpg.head())

# subtract the mean mpg in 2008 from that in 2018 to get the change in mpg
model_mpg['mpg_change'] = model_mpg['cmb_mpg'] - model_mpg['cmb_mpg_2008']

print(model_mpg.head())

# Find the max mpg change
max_change = model_mpg['mpg_change'].max()
print(max_change)

# and add it to the column
print(model_mpg[model_mpg['mpg_change'] == max_change])

#  find the index of the row containing a column's maximum value
idx = model_mpg.mpg_change.idxmax()
print(idx)

print(model_mpg.loc[idx])
