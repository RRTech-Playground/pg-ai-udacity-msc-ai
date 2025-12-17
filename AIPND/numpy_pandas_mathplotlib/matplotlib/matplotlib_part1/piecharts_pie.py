import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#%matplotlib inline

pokemon = pd.read_csv('pokemon.csv')
#print(pokemon.shape)
#print(pokemon.head(10))

sorted_counts = pokemon['generation_id'].value_counts()
plt.pie(sorted_counts, labels = sorted_counts.index, startangle = 90, counterclock = False)
print(sorted_counts.index)

# We have the used option `Square`.
# Though, you can use either one specified here -
# https://matplotlib.org/api/_as_gen/matplotlib.pyplot.axis.html?highlight=pyplot%20axis#matplotlib-pyplot-axis
plt.axis('square')

plt.show()
