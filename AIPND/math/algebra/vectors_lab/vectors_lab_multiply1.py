import numpy as np
import matplotlib.pyplot as plt
#%matplotlib inline

# Define vector v
v = np.array([1, 1])

# Plots vector v as blue arrow with red dot at origin (0,0) using Matplotlib

# Creates axes of plot referenced 'ax'
ax = plt.axes()

# Plots red dot at origin (0,0)
ax.plot(0, 0, 'ro')

# Plots vector v as blue arrow starting at origin 0,0
ax.arrow(0, 0, *v, color='b', linewidth=2.5, head_width=0.3, head_length=0.35)

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