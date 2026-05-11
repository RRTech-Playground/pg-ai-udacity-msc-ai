# import pandas and read the two data csvs
import pandas as pd


# load 2008 and 2018 datasets
df_08 = pd.read_csv("data/clean_08.csv")
df_18 = pd.read_csv("data/clean_18.csv")


#######################
# Optimization

df = df_08
print(df.info())

# Numerical optimization
df.city_mpg.value_counts()
df.hwy_mpg.value_counts()
df.cmb_mpg.value_counts()

# Change city_mpg, hwy_mpg, cmb_mpg to be an int using .astype()
df["city_mpg"] = df["city_mpg"].astype("int")
df["hwy_mpg"] = df["hwy_mpg"].astype("int")
df["cmb_mpg"] = df["cmb_mpg"].astype("int")

print(df.info())

# Well that did not change anything. Instead of an int64, let's change them to be int8. The values for each column only range from 8 - 48. Use .describe() to view the min/max of each column
df[["city_mpg", "hwy_mpg", "cmb_mpg"]].describe()

# Change the data type to be an int8
df["city_mpg"] = df["city_mpg"].astype("int8")
df["hwy_mpg"] = df["hwy_mpg"].astype("int8")
df["cmb_mpg"] = df["cmb_mpg"].astype("int8")

print(df.info())

# Now we are getting somewhere! We just changed the memory usage from 100.4+ KB to 80.1+ KB by changing how we are storing our int values.

# String Optimization

# Look at the value counts of each object data type: trans, drive, fuel, veh_class, smartway, and model.
df["trans"].value_counts()
df["drive"].value_counts()
df["fuel"].value_counts()
df["veh_class"].value_counts()
df["smartway"].value_counts()
df["model"].value_counts()

# Except for model, all of the object types have 2 - 13 unique values.

# In pandas there is a specialized data type called Categorical. Categorical data types are useful when you have object columns with a low number of unique values.

# assign trans, drive, fuel, veh_class, and smartway to "category" using .astype()
df["trans"] = df["trans"].astype("category")
df["drive"] = df["drive"].astype("category")
df["fuel"] = df["fuel"].astype("category")
df["veh_class"] = df["veh_class"].astype("category")
df["smartway"] = df["smartway"].astype("category")

print(df.info())

# Wow! By changing those columns to categories, we further reduced our dataset from 80.1+ KB to 47.8+ KB. We effectively reduced the memory usage by 50%.


#######################
# Concatenation

# Create two arrays as long as the number of rows in the 2008 and 2018 dataframes.
# Note: If you had different values for each row in the new column, you'd have to assign the column an array the exact size as the DataFrame, but because we are only using one value, this is a nice little trick!

df_08['year'] = '2008'
print(df_08.head())

df_18['year'] = '2018'
print(df_18.head())

# concat dataframes
df = pd.concat([df_08, df_18])

# view dataframe to check for success
print(df.head(1))
print(df.tail(1))

# safe concatenated data set
df.to_csv("data/clean_08_18_concatenated.csv", index=False)
