import pandas as pd
import matplotlib.pyplot as plt

#######################
# Histogram and Bar Charts

df_census = pd.read_csv('data/census_income_data.csv')
print(df_census.info())

# Histogram of all columns
df_census.hist(figsize=(8, 8))
#plt.show()

# Histogram of the age column
df_census['age'].hist()
#plt.show()

# Bar chart of education
df_census['education'].value_counts().plot(kind='bar')
#plt.show()

# Histogram of the age column using plot()
df_census['age'].plot(kind='hist')
#plt.show()

#######################
# Scatter and Box Plots

df_cancer = pd.read_csv('data/cancer_data_edited.csv')
print(df_cancer.info())

# Scatter plot of all columns
pd.plotting.scatter_matrix(df_cancer, figsize=(15, 15))
#plt.show()

# Scatter plot of two columns, compactness and concavity
df_cancer.plot(x='compactness', y='concavity', kind='scatter')
#plt.show()

# Box plot of concave_points
df_cancer['concave_points'].plot(kind='box')
#plt.show()


#######################
# Visuals Quiz

# load powerplant_data_edited.csv
df = pd.read_csv('data/powerplant_data_edited.csv')
print(df.head())

# plot relationship between temperature and electrical output using a scatter plot
df.plot(x='temperature', y='energy_output', kind='scatter')
#plt.show()

# plot distribution of humidity using a histogram
df['humidity'].hist()
#plt.show()

# plot box plots for temperature
df['temperature'].plot(kind='box')
#plt.show()

# plot box plots for exhaust_vacuum
df['exhaust_vacuum'].plot(kind='box')
#plt.show()

# plot box plots for pressure
df['pressure'].plot(kind='box')
#plt.show()

# plot box plots for humidity
df['humidity'].plot(kind='box')
#plt.show()

# plot box plots for energy_output
df['energy_output'].plot(kind='box')
#plt.show()

