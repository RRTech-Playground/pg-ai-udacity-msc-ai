import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
#%matplotlib inline

# Read the CSV file
fuel_econ = pd.read_csv('fuel_econ.csv')
fuel_econ.head(10)

# Resize figure to accommodate two plots
#plt.figure(figsize = [20, 5])

# PLOT ON LEFT - SIMPLE SCATTER
#plt.subplot(1, 2, 1)
#sb.regplot(data = fuel_econ, x = 'year', y = 'comb', truncate=False)

# PLOT ON RIGHT - SCATTER PLOT WITH JITTER
#plt.subplot(1, 2, 2)
# In the sb.regplot() function below, the `truncate` argument accepts a boolean.
# If truncate=True, the regression line is bounded by the data limits.
# Else if truncate=False, it extends to the x axis limits.
# The x_jitter will make each x value will be adjusted randomly by +/-0.3
#sb.regplot(data = fuel_econ, x = 'year', y = 'comb', truncate=False, x_jitter=0.3)

# The scatter_kws helps specifying the opaqueness of the data points.
# The alpha take a value between [0-1], where 0 represents transparent, and 1 is opaque.
sb.regplot(data = fuel_econ, x = 'year', y = 'comb', truncate=False, x_jitter=0.3, scatter_kws={'alpha':1/20})

# Alternative way to plot with the transparency.
# The scatter() function below does NOT have any argument to specify the Jitter
plt.scatter(data = fuel_econ, x = 'year', y = 'comb', alpha=1/20)

plt.show()
