import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb

pokemon = pd.read_csv('pokemon.csv')
print(pokemon.shape)
print(pokemon.head(10))

sb.countplot(data=pokemon, x='generation_id')

plt.show()