import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
#%matplotlib inline

pokemon = pd.read_csv('pokemon.csv')
#print(pokemon.shape)
#print(pokemon.head(10))

# The pokemon dataset is available to download at the bottom of this page. # The kind argument can take any one value from {“hist”, “kde”, “ecdf”}.
#sb.displot(pokemon['speed'], kind='hist')

# Use the 'kde' kind for kernel density estimation
sb.displot(pokemon['speed'], kind='kde')

plt.show()