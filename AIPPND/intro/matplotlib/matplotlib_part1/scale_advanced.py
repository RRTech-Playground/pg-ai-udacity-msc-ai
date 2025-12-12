import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

pokemon = pd.read_csv('pokemon.csv')

def sqrt_trans(x, inverse=False):
    """transformation helper function"""
    if not inverse:
        return np.sqrt(x)
    else:
        return x**2

    ## Bin resizing, to transform the x-axis
bin_edges = np.arange(0, sqrt_trans(pokemon['weight'].max()) + 1, 1)

## Plot the scaled data
plt.hist(pokemon['weight'].apply(sqrt_trans), bins=bin_edges)

## Identify the tick-locations
tick_locs = np.arange(0, sqrt_trans(pokemon['weight'].max()) + 10, 10)

## Apply x-ticks
plt.xticks(tick_locs, sqrt_trans(tick_locs, inverse=True).astype(int))

plt.show()


