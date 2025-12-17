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
# Refer - https://pandas.pydata.org/pandas-docs/version/2.1.3/reference/api/pandas.CategoricalDtype.html
vclasses = pd.api.types.CategoricalDtype(ordered=True, categories=sedan_classes)

# Use pandas.astype() to convert the "VClass" column from a plain object type into an ordered categorical type
fuel_econ['VClass'] = fuel_econ['VClass'].astype(vclasses)

# The existing trans column has multiple sub-types of Automatic and Manual.
# But, we need plain two types, either Automatic or Manual. Therefore, add a new column.
# The Series.apply() method invokes the lambda function on each value of trans column.
# In python, a lambda function is an anonymous function that can have only one expression.
fuel_econ['trans_type'] = fuel_econ['trans'].apply(lambda x: x.split()[0])
#print(fuel_econ.head())

# Use group_by() and size() to get the number of cars and each combination of the two variable levels as a pandas Series
ct_counts = fuel_econ.groupby(['VClass', 'trans_type']).size()
# Number of cars in each vehicle type and transmission combination
#print(ct_counts)

# Use Series.reset_index() to convert a series into a dataframe object
ct_counts = ct_counts.reset_index(name='count')
# A DataFrame object created from the Series generated in the step above
#print(ct_counts)

# Use DataFrame.pivot() to rearrange the data, to have vehicle class on rows
ct_counts = ct_counts.pivot(index='VClass', columns='trans_type', values='count')
# The DataFrame to plot on heatmap
#print(ct_counts)

#sb.heatmap(ct_counts)
sb.heatmap(ct_counts, annot=True, fmt='d')

# Use fmt='.0f' to show NaN columns if needed (not needed here).
#sb.heatmap(ct_counts, annot=True, fmt='.0f')

plt.show()