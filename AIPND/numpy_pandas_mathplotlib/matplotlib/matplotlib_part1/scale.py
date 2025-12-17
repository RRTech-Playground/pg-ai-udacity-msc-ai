import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

pokemon = pd.read_csv('pokemon.csv')
print(pokemon.head(10))

plt.figure(figsize = [20, 5])

## HISTOGRAM ON LEFT: full data without scaling
plt.subplot(1, 2, 1)
plt.hist(data=pokemon, x='weight')
## Display a label on the x-axis
plt.xlabel('Initial plot with original data')

## HISTOGRAM ON RIGHT
plt.subplot(1, 2, 2)

## Get the ticks for bins between [0 - maximum weight]
bins = np.arange(0, pokemon['weight'].max() + 40, 40)
plt.hist(data=pokemon, x='weight', bins=bins)

## The argument in the xscale() represents the axis scale type to apply.
## The possible values are: {"linear", "log", "symlog", "logit", ...}
## Refer - https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.pyplot.xscale.html
plt.xscale('log')
plt.xlabel('The x-axis limits NOT are changed. They are only scaled to log-type')

plt.show()