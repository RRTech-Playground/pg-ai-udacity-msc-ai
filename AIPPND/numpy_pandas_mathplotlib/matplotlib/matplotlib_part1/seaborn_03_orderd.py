import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb

pokemon = pd.read_csv('pokemon.csv')
print(pokemon.shape)
print(pokemon.head(10))

## Ordered
freq = pokemon['generation_id'].value_counts()

# Get the indexes of the Series
gen_order = freq.index

# Plot the bar chart in the decreasing order of the frequency of the `generation_id`
sb.countplot(data=pokemon, x='generation_id', order=gen_order)

plt.show()