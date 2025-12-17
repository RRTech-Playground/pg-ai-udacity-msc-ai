import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#%matplotlib inline

pokemon = pd.read_csv('pokemon.csv')
#print(pokemon.shape)
#print(pokemon.head(10))

# Resize the chart, and have two plots side-by-side set a larger figure size for subplots
plt.figure(figsize = [20, 5])

# histogram on left, example of too-large bin size 1 row, 2 cols, subplot 1
plt.subplot(1, 2, 1) # 1 row, 2 cols, subplot 1
bins = np.arange(0, pokemon['speed'].max()+4, 4)
plt.hist(data = pokemon, x = 'speed', bins = bins)

# histogram on right, example of too-small bin size
plt.subplot(1, 2, 2) # 1 row, 2 cols, subplot 2
#bins = np.arange(0, pokemon['speed'].max()+1/4, 1/4)
#plt.hist(data = pokemon, x = 'speed', bins = bins)

plt.show()