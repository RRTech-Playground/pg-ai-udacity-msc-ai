import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
#%matplotlib inline

# Read the CSV file
fuel_econ = pd.read_csv('fuel_econ.csv')
fuel_econ.head(10)

# Scatter plot
#plt.scatter(data=fuel_econ, x='displ', y='comb')  # matplotlib.pyplot.scatter()

#sb.regplot(data=fuel_econ, x='displ', y='comb')  # seaborn.regplot()
#sb.regplot(data=fuel_econ, x='displ', y='comb', fit_reg = False)  # turn off regression line

# Example of using log and putting the data directly on the plot
def log_trans(x, inverse=False):
    if not inverse:
        return np.log10(x)
    else:
        return np.power(10, x)

sb.regplot(data=fuel_econ, x=fuel_econ['displ'], y=fuel_econ['comb'].apply(log_trans))

tick_locs = [10, 20, 50, 100]
plt.yticks(log_trans(tick_locs), tick_locs)


plt.xlabel('Displacement (1)')
plt.ylabel('Combined Fuel Eff. (mpg)')

plt.show()