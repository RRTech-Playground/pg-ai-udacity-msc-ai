import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
#%matplotlib inline

# Read the CSV file
fuel_econ = pd.read_csv('fuel_econ.csv')
fuel_econ.head(10)

# Types of sedan cars
sedan_classes = ['Minicompact Cars', 'Subcompact Cars', 'Compact Cars', 'Midsize Cars', 'Large Cars']

# Returns the types for sedan_classes with the categories and orderedness
# Refer - https://pandas.pydata.org/pandas-docs/version/0.23.4/generated/pandas.api.types.CategoricalDtype.html
vclasses = pd.api.types.CategoricalDtype(ordered=True, categories=sedan_classes)

print(vclasses)

# Use pandas.astype() to convert the "VClass" column from a plain object type into an ordered categorical type
fuel_econ['VClass'] = fuel_econ['VClass'].astype(vclasses)

# Violin plot
#sb.violinplot(data=fuel_econ, x='VClass', y='comb')

# Without showing the mean
#sb.violinplot(data=fuel_econ, x='VClass', y='comb', inner=None)

# Showing the quartiles of the distribution instead of the mean
sb.violinplot(data=fuel_econ, x='VClass', y='comb', inner='quartiles')

# Rotate the x-axis labels
plt.xticks(rotation=15)

# Change the orientation of the violin plot
#sb.violinplot(data=fuel_econ, x='comb',  y='VClass', inner=None)

plt.show()