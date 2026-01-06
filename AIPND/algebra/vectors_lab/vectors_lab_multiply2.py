import numpy as np
import matplotlib.pyplot as plt
#%matplotlib inline

# Define vector v
v = np.array([1, 1])

# Define scalar a
a = 3

# DONE 1.: Define vector av - as vector v multiplied by a
av = a * v

# Plots vector v as blue arrow with red dot at origin (0,0) using Matplotlib

# Creates axes of plot referenced 'ax'
ax = plt.axes()

# Plots red dot at origin (0,0)
ax.plot(0, 0, 'ro')

# Plots vector v as blue arrow starting at origin 0,0
ax.arrow(0, 0, *v, color='b', linewidth=2.5, head_width=0.3, head_length=0.35)

# DONE 2.: Plot vector av as dotted (linestyle='dotted') vector of cyan color (color='c')
# using ax.arrow() statement above as template for the plot
ax.arrow(0, 0, *av, color='c', linestyle='dotted', linewidth=2.5, head_width=0.3, head_length=0.35)

# Sets limit for plot for x-axis
plt.xlim(-2, 4)

# Set major ticks for x-axis
major_xticks = np.arange(-2, 4)
ax.set_xticks(major_xticks)

# Sets limit for plot for y-axis
plt.ylim(-1, 4)

# Set major ticks for y-axis
mayor_yticks = np.arange(-1, 4)
ax.set_yticks(mayor_yticks)

# Creates gridlines for only major tick marks
plt.grid(visible=True, which='major')

# Display final plot
plt.show()