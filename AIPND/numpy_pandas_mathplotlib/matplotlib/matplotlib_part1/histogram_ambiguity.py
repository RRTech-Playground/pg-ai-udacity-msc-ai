import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
#%matplotlib inline

die_rolls = pd.read_csv('die-rolls.csv')

# A fair dice has six-faces having numbers [1-6]. # There are 100 dices, and two trials were conducted. # In each trial, all 100 dices were rolled down, and the outcome [1-6] was recorded.

# The `Sum` column represents the sum of the outcomes in the two trials, for each given dice.

print(die_rolls.head(10))

plt.figure(figsize = [20, 5])

# Histogram on the left, bin edges on integers
plt.subplot(1, 2, 1)
bin_edges = np.arange(2, 12 + 1.1, 1) # note `+1.1`, see below
plt.hist(data=die_rolls, x='Sum', bins=bin_edges)
plt.xticks(np.arange(2, 12 + 1, 1))

# Histogram on the right, bin edges between integers
plt.subplot(1, 2, 2)
bin_edges = np.arange(1.5, 12.5 + 1, 1)
plt.hist(data=die_rolls, x='Sum', bins=bin_edges, rwidth=0.7)
plt.xticks(np.arange(2, 12 + 1, 1))

# Using rwidth to show the bars in the histogram. But be careful to use, as it could change the perception of the data!
# bin_edges = np.arange(1.5, 12.5 + 1, 1)
# plt.hist(data=die_rolls, x='Sum', bins=bin_edges, rwidth=0.7)
# plt.xticks(np.arange(2, 12 + 1, 1))

plt.show()


# Define the figure size
plt.figure(figsize = [20, 5])

# histogram on left: full data
plt.subplot(1, 2, 1)
bin_edges = np.arange(0, pokemon['height'].max()+0.5, 0.5)
plt.hist(data=pokemon, x='height', bins = bin_edges)

# histogram on right: focus in on bulk of data < 6
plt.subplot(1, 2, 2)
bin_edges = np.arange(0, pokemon['height'].max()+0.2, 0.2)
plt.hist(data=pokemon, x='height', bins = bin_edges)
plt.xlim(0, 6) # could also be called as plt.xlim((0, 6))