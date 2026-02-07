import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#%matplotlib inline

pokemon = pd.read_csv('pokemon.csv')
pokemon.head(10)

# Define the figure size
plt.figure(figsize = [20, 5])

plt.subplot(1, 2, 1)
# Get the ticks for bins between [0-15], at an interval of 0.5
bins = np.arange(0, pokemon['height'].max()+0.5, 0.5)
# Plot the histogram for the height column
plt.hist(data=pokemon, x='height', bins=bins)

plt.subplot(1, 2, 2)
# Get the ticks for bins between [0-15], at an interval of 0.5
bins = np.arange(0, pokemon['height'].max()+0.2, 0.2)
plt.hist(data=pokemon, x='height', bins=bins)

# Set the upper and lower bounds of the bins that are displayed in the plot
# Refer here for more information - https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.pyplot.xlim.html
# The argument represent a tuple of the new x-axis limits.
plt.xlim((0,6))

plt.show()