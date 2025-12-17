import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

pokemon = pd.read_csv('pokemon.csv')
print(pokemon.shape)
print(pokemon.head(10))

x = pokemon['generation_id'].unique()
y = pokemon['generation_id'].value_counts(sort=False)

plt.bar(x, y)

plt.xlabel('generation_id')
plt.ylabel('count')

plt.show()