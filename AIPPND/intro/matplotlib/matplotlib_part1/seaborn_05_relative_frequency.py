import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb

from matplotlib import rcParams

pokemon = pd.read_csv('pokemon.csv')
print(pokemon.shape)
print(pokemon.head(10))

## Step 1, Data wrangling
pkmn_types = pokemon.melt(id_vars=['id', 'species'],
                          value_vars=['type_1', 'type_2'],
                          var_name='type_level',
                          value_name='type')

print(pkmn_types.head(10))


## Step 2, find the frequency of unique values in the type column

# Count the frequency of unique values in the `type` column of pkmn_types dataframe.
# By default, returns the decreasing order of the frequency.
type_counts = pkmn_types['type'].value_counts()
print(type_counts)

# Get the unique values of the type column, in the decreasing order of the frequency.
type_order = type_counts.index
print(type_order)


## Step 3, Plot the horizontal bar charts
#base_color = sb.color_palette()[0]
#sb.countplot(data=pkmn_types, y='type', color=base_color, order=type_order)


## Step 4, Find the maximum proportion of bar

# Returns the sum of all not-null values in type column n_pokemon
n_pokemon = pkmn_types['type'].value_counts().sum()

# Return the highest frequency in the type column
max_type_count = type_counts[0]

# Return the maximum proportion, or in other words, # compute the length of the longest bar in terms of the proportion
max_prop = max_type_count / n_pokemon
print(max_prop)


## Step 5, Create an array of evenly spaced proportioned values. Use numpy.arange() function to produce a set of evenly spaced proportioned values between 0 and max_prop, with a step size 2\%
tick_props = np.arange(0, max_prop, 0.02)
print(tick_props)


## Step 6, Create a list of String values that can be used as tick labels.
# Use a list comprehension to create tick_names that we will apply to the tick labels. # Pick each element `v` from the `tick_props`, and convert it into a formatted string. # `{:0.2f}` denotes that before formatting, we 2 digits of precision and `f` is used to represent floating point number. # Refer [here](https://docs.python.org/2/library/string.html#format-string-syntax) for more details
tick_names = ['{:0.2f}'.format(v) for v in tick_props]
print(tick_names)


## Step 7, Plot the bar chart, with new x-tick labels
sb.countplot(data=pkmn_types, y='type', order=type_order)
# Change the tick locations and labels
plt.xticks(tick_props * n_pokemon, tick_names)
plt.xlabel('proportion')


## Step 8, Print the text (proportion) on the bars of a horizontal plot
# Considering the same chart from the Example 1 above, print the text (proportion) on the bars
base_color = sb.color_palette()[0]
sb.countplot(data=pkmn_types, y='type', color=base_color, order=type_order)

# Logic to print the proportion text on the bars
for i in range (type_counts.shape[0]):
    # Remember, type_counts contains the frequency of unique values in the type column in decreasing order.
    count = type_counts[i]
    # Convert count into a percentage, and then into string
    pct_string = '{:0.1f}'.format(100*count/n_pokemon)
    # Print the string value on the bar.
    # Read more about the arguments of text() function [here](https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.pyplot.text.html)
    plt.text(count+1, i, pct_string, va='center')


## Step 9, Print the text (proportion) below the bars of a Vertical plot.

# Clear the current figure to reset the axes and ticks
plt.clf()

# Considering the same chart from the Example 1 above, print the text (proportion) BELOW the bars
base_color = sb.color_palette()[0]
sb.countplot(data=pkmn_types, x='type', color=base_color, order=type_order)

# Recalculating the type_counts just to have clarity.
type_counts = pkmn_types['type'].value_counts()

# get the current tick locations and labels
locs, labels = plt.xticks(rotation=90)

# loop through each pair of locations and labels
for loc, label in zip(locs, labels):

    # get the text property for the label to get the correct count
    count = type_counts[label.get_text()]
    pct_string = '{:0.1f}%'.format(100*count/n_pokemon)

    # print the annotation just below the top of the bar
    plt.text(loc, count+2, pct_string, ha = 'center', color = 'black')


## Step 10, Changing the size of the plot
from matplotlib import rcParams
# Specify the figure size in inches, for both X, and Y axes
rcParams['figure.figsize'] = 12,4

# Show the plot
plt.show()