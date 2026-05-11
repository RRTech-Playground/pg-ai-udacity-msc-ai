import pandas as pd
from matplotlib import pyplot as plt

# Load the dataset
df_census = pd.read_csv('data/census_income_data.csv')

# Create two new DataFrames that are separated on people who make more or less than 50K
df_a = df_census[df_census['income'] == ' >50K']
df_b = df_census[df_census['income'] == ' <=50K']

# Get the value counts of education, we need the index to keep the order consistent for
ind = df_a['education'].value_counts().index
#df_a['education'].value_counts()[ind].plot(kind='bar')
#plt.show()

# Use the same index to keep the order consistent
#df_b['education'].value_counts()[ind].plot(kind='bar')
#plt.show()

# Get the value counts of workclass and create a pie chart for each group.
# Use an index again to keep values in the same order
ind = df_a['workclass'].value_counts().index
#df_a['workclass'].value_counts()[ind].plot(kind='pie', figsize=(8,8))
#plt.show()

#df_b['workclass'].value_counts()[ind].plot(kind='pie', figsize=(8,8))
#plt.show()

# Create histograms of age for each group
#df_a['age'].hist()
#plt.show()
print(df_a['age'].describe())

df_b['age'].hist()
plt.show()
print(df_b['age'].describe())



