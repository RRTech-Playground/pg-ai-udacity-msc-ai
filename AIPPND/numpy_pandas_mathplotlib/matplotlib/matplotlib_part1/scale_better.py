import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

pokemon = pd.read_csv('pokemon.csv')
print(pokemon.head(10))

## Describe the data
print(pokemon['weight'].describe())

## Transform the describe() to a scale of log10
## Documentation: [numpy `log10`](https://docs.scipy.org/doc/numpy/reference/generated/numpy.log10.html)
print(np.log10(pokemon['weight'].describe()))

plt.figure(figsize = [20, 5])

## HISTOGRAM ON LEFT: without optimization of the ticks
plt.subplot(1, 2, 1)

## Axis transformation ## Bin size
bins = 10 ** np.arange(-1, 3 + 0.1, 0.1)
plt.hist(data=pokemon, x='weight', bins=bins)

## The argument in the xscale() represents the axis scale type to apply. ## The possible values are: {"linear", "log", "symlog", "logit", ...}
plt.xscale('log')

## Apply x-axis label ## Documentatin: [matplotlib `xlabel`](https://matplotlib.org/api/_as_gen/matplotlib.pyplot.xlabel.html))
plt.xlabel('x-axis limits are changed, and scaled to log-type')

## HISTOGRAM ON RIGHT: also with optimization of the ticks
plt.subplot(1, 2, 2)

## Get the ticks for bins between [0 - maximum weight]
bins = 10 ** np.arange(-1, 3 + 0.1, 0.1)

## Generate the x-ticks you want to apply
ticks = [0.1, 0.3, 1, 3, 10, 30, 100, 300, 1000]
## Convert ticks into string values, to be displaye dlong the x-axis
labels = ['{}'.format(v) for v in ticks]

## Plot the histogram
plt.hist(data=pokemon, x='weight', bins=bins)

## The argument in the xscale() represents the axis scale type to apply. ## The possible values are: {"linear", "log", "symlog", "logit", ...}
plt.xscale('log')

## Apply x-ticks
plt.xticks(ticks, labels)

plt.xlabel('same as left, but ticks manually adjusted')

plt.show()