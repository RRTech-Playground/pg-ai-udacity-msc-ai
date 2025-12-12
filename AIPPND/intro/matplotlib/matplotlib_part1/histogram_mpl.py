import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#%matplotlib inline

pokemon = pd.read_csv('pokemon.csv')
#print(pokemon.shape)
#print(pokemon.head(10))

#print(pokemon['speed'])

#plt.hist(data=pokemon, x='speed')
#plt.hist(data=pokemon, x='speed', bins=20)

# Create bins with a step size of 5
bins = np.arange(0, pokemon['speed'].max() + 5, 5)

# Plot a histogram using the defined bins
plt.hist(data=pokemon, x='speed', bins=bins)

plt.show()