import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#%matplotlib inline

pokemon = pd.read_csv('pokemon.csv')
#print(pokemon.shape)
#print(pokemon.head(10))

sorted_counts = pokemon['generation_id'].value_counts()

plt.pie(sorted_counts, labels = sorted_counts.index, startangle = 90, counterclock = False, wedgeprops = {'width' : 0.4});

plt.axis('square')

plt.show()

