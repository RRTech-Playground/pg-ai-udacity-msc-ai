import pandas as pd
from matplotlib import pyplot as plt

# load datasets
df_08 = pd.read_csv('data/clean_08.csv')
df_18 = pd.read_csv('data/clean_18.csv')

print(df_08.head(1))

## Q1: Are more unique models using alternative sources of fuel? By how much?

# Let's first look at what the sources of fuel are and which ones are alternative sources.
print(df_08.fuel.value_counts())
print(df_18.fuel.value_counts())

# how many unique models used alternative sources of fuel in 2008
alt_08 = df_08.query('fuel in ["CNG", "ethanol"]').model.nunique()
print(alt_08)

# how many unique models used alternative sources of fuel in 2018
alt_18 = df_18.query('fuel in ["Ethanol", "Electricity"]').model.nunique()
print(alt_18)

# Create a bar chart with both 2008 and 2018 alternative source numbers
pd.DataFrame(
    {"year": ["2008", "2018"], "model_num": [alt_08, alt_18]}
).plot(
    kind="bar",
    x="year",
    title="Number of Unique Models Using Alternative Fuels",
    legend=False,
    xlabel="Year",
    ylabel="Number of Unique Models",
    rot=0
)

# total unique models each year
total_08 = df_08.model.nunique()
total_18 = df_18.model.nunique()
print(total_08, total_18)

# Find the proportion of alternative models by the total number of models for 2008 and 2018
prop_08 = alt_08 / total_08
prop_18 = alt_18 / total_18
print(prop_08, prop_18)

# Create a bar chart with both 2008 and 2018 proportion values
pd.DataFrame(
    {"year": ["2008", "2018"], "model_num": [prop_08, prop_18]}
).plot(
    kind="bar",
    x="year",
    title="Proportion of Unique Models Using Alternative Fuels",
    legend=False,
    xlabel="Year",
    ylabel="Proportion of Unique Models",
    rot=0
)


## Q2: How much have vehicle classes improved in fuel economy?

# Let's look at the average fuel economy for each vehicle class for both years.

# group by veh_class and find the mean of cmb_mpg
veh_08 = df_08.groupby('veh_class').cmb_mpg.mean()
print(veh_08)

veh_18 = df_18.groupby('veh_class').cmb_mpg.mean()
print(veh_18)

# Find how much they've increased by for each vehicle class
# Take the difference of veh_18 and veh_08 to find the increase
inc = veh_18 - veh_08
print(inc)

# Drop any NaN values to only plot the classes that exist in both years
inc.dropna(inplace=True)
# Create a bar chart to show the increase for vehicle types
inc.plot(
    kind="bar",
    x="year",
    title="Improvements in Fuel Economy from 2008 to 2018 by Vehicle Class",
    legend=False,
    xlabel="Vehicle Class",
    ylabel="Increase in Average Combined MPG",
    rot=0,
    figsize=(8, 5)
)


## Q3: What are the characteristics of SmartWay vehicles? Have they changed over time?

# We can analyze this by filtering each dataframe by SmartWay classification and exploring these datasets.

# smartway labels for 2008
print(df_08.smartway.unique())

# get all smartway vehicles in 2008
smart_08 = df_08.query('smartway == "yes"')
print(smart_08.describe())

# Use what you've learned so for to further explore this dataset on 2008 smartway vehicles.

# smartway labels for 2018
df_18.smartway.unique()

# get all smartway vehicles in 2018
smart_18 = df_18.query('smartway in ["Yes", "Elite"]')
print(smart_18.describe())

# Create a bar chart to find average cmb_mpg of smartway vehicles
pd.DataFrame(
    {"year": ["2008", "2018"], "model_num": [smart_08["cmb_mpg"].mean(), smart_18["cmb_mpg"].mean()]}
).plot(
    kind="bar",
    x="year",
    title="Average Combined MPG from Smartway Vehicles from 2008 and 2018",
    legend=False,
    xlabel="Year",
    ylabel="Avg cmb_mpg",
    rot=0
)


## Q4: What features are associated with better fuel economy?

# You can explore trends between cmb_mpg and the other features in this dataset, or filter this dataset like in the previous question and explore the properties of that dataset. For example, you can select all vehicles that have the top 50% fuel economy ratings like this.

# Create the top 50% by finding cmb_mpg >= the average cmb_mpg
# Create the bottom 50% by finding cmb_mpg < the average cmb_mpg
top_08 = df_08.query('cmb_mpg >= cmb_mpg.mean()')
bottom_08 = df_08.query('cmb_mpg < cmb_mpg.mean()')
print(top_08.describe())

# Create the top 50% by finding cmb_mpg >= the average cmb_mpg
# Create the bottom 50% by finding cmb_mpg < the average cmb_mpg
top_18 = df_18.query('cmb_mpg >= cmb_mpg.mean()')
bottom_18 = df_18.query('cmb_mpg < cmb_mpg.mean()')
print(top_18.describe())

# Plot a bar chart showing top and bottom metrics for 2008 and 2018
# Make sure to use numeric_only=True when geting the mean as we only
# want features that are numbers
pd.DataFrame(
    {
        "Bottom 2008": bottom_08.mean(numeric_only=True),
        "Bottom 2018": bottom_18.mean(numeric_only=True),
        "Top 2008": top_08.mean(numeric_only=True),
        "Top 2018": top_18.mean(numeric_only=True),
    },
    index=top_18.mean(numeric_only=True).index
).plot(
    kind="bar",
    title="Average Vehicle Features Separated by 50% Above or Below cmb_mpg",
    legend=True,
    xlabel="Feature",
    ylabel="Average value of feature",
    rot=0,
    figsize=(12, 8)
)

plt.show()