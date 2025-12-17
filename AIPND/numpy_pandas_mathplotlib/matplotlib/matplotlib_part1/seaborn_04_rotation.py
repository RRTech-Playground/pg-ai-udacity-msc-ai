import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb

pokemon = pd.read_csv('pokemon.csv')
print(pokemon.shape)
print(pokemon.head(10))

## Rotate the category labels
#sb.countplot(data=pokemon, x='type_1')

# Use xticks to rotate the category labels (not axes) counter-clockwise
#plt.xticks(rotation=90)


## Rotate the axes clockwise
type_order = pokemon['type_1'].value_counts().index
sb.countplot(data=pokemon, y='type_1', order=type_order)

plt.show()