import pandas as pd

df = pd.read_csv("data/census_income_data.csv")

#######################
# Groupby

# view the columns available
print(df.columns)

# calculate the mean values for all numeric columns
print(df.mean(numeric_only=True))

# groupby "workclass" to see the different mean values for all numeric columns
print(df.groupby("workclass").mean(numeric_only=True))

# groupby "workclass" and "race" to see the different mean values for all numeric columns
print(df.groupby(["workclass", "race"]).mean(numeric_only=True))

# set as_index=False to keep "workclass" and "race" as non index values
# select just "capital-gain" to view only that column's mean
print(df.groupby(["workclass", "race"], as_index=False)["capital-gain"].mean())


#######################
# Summation

df_census = pd.read_csv("data/census_income_data.csv")

print(df_census.head())

# How much capital was gained and lost in our dataset?
print(df_census[["capital-gain", "capital-loss"]].sum())

# What are the different workclass types?
print(df_census["workclass"].value_counts())

# If we group by 'workclass'
print(df_census.groupby(by="workclass").sum(numeric_only=True))

# Let's do a similar comparison for occupation
print(df_census["occupation"].value_counts())

# Group by and take the sum for occupation
print(df_census.groupby(by="occupation").sum(numeric_only=True).sort_values(by="hours-per-week", ascending=False))

# Let's do a similar comparison for marital status
print(df_census["marital-status"].value_counts())

# Group by and take the sum for marital-status
print(df_census.groupby(by="marital-status").sum(numeric_only=True).sort_values(by="hours-per-week", ascending=False))


#######################
# Measure of Center

# What was the average capital gained and lost in our dataset?
print(df_census[["capital-gain", "capital-loss"]].mean())

# What are the different workclass types
print(df_census["workclass"].value_counts())

# If we group by 'workclass', what are some interesting questions and answers?
print(df_census.groupby(by="workclass").mean(numeric_only=True))
print(df_census.groupby(by="workclass").median(numeric_only=True))

# How about for occupation?
print(df_census["occupation"].value_counts())
print(df_census.groupby(by="occupation").mean(numeric_only=True))
print(df_census.groupby(by="occupation").median(numeric_only=True))


#######################
# Measure of Spread

# What was the 25%, 50%, 75%, 90%, and 95% quantile of capital gained and lost in our dataset?
print(df_census[["capital-gain", "capital-loss"]].quantile(q=[0.25, 0.5, 0.75, 0.9, 0.95]))

## Using Quantile, Standard Deviation, and Variance on a Group

# What are the different workclass types
print(df_census["workclass"].value_counts())

# Standard deviation is useful when paired with mean, as it will tell you how distributed you data is in relation to the mean
print(df_census.groupby(by="workclass")[["age", "capital-gain", "capital-loss", "hours-per-week"]].mean())
print(df_census.groupby(by="workclass")[["age", "capital-gain", "capital-loss", "hours-per-week"]].std())

# Variance can be used to help spot distributions with outliers
print(df_census.groupby(by="workclass")[["age", "capital-gain", "capital-loss", "hours-per-week"]].var())

# Remember that we can use the `.describe()` method on groups to produce many of these measurements.
# We can also use the `percentiles` parameter to customize which quantiles we want to target.
print(df_census.groupby(by="workclass")[["capital-gain", "capital-loss"]].describe(percentiles=[0.5, 0.9, 0.95]))
