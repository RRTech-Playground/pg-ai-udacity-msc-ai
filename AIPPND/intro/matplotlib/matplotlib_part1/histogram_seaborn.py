import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
#%matplotlib inline

pokemon = pd.read_csv('pokemon.csv')
#print(pokemon.shape)
#print(pokemon.head(10))

# Plot a histogram with KDE (Kernel Density Estimate) line
#sb.displot(data=pokemon, x='speed', kde=True, stat='density', bins=20)

# Plot a histogram without the KDE line
#sb.histplot(data=pokemon, x='speed')

# Define bin edges for the histogram based on the speed column
bin_edges = np.arange(0, pokemon['speed'].max() + 1, 5)

# Plot the distribution of the speed column using
sb.histplot(data=pokemon, x='speed', bins=bin_edges, alpha=1 )

plt.show()