# Import NumPy and Matplotlib - only needed for this lab solution notebook
import numpy as np
import matplotlib.pyplot as plt

# Define vector v
v = np.array([-1, 2])

# DONE 1.: Define vector i_hat as transformed vector i_hat(ihat_t)
# where x=3 and y=1 instead of x=1 and y=0
ihat_t = np.array([3, 1])

# DONE 2.: Define vector j_hat as transformed vector j_hat(jhat_t)
# where x=1 and y=2 instead of x=0 and y=1
jhat_t = np.array([1, 2])

# Define v_ihat_t - as v[0](x) multiplied by transformed vector ihat
v_ihat_t = v[0] * ihat_t

# Define v_jhat_t - as v[1](y) multiplied by transformed vector jhat
v_jhat_t = v[1] * jhat_t

# DONE 3.: Define transformed vector v (v_t) as
# vector v_ihat_t added to vector v_jhat_t
v_t = v_ihat_t + v_jhat_t


# Plot that graphically shows vector v (color='skyblue') can be transformed
# into transformed vector v (v_trfm - color='b') by adding v[0]*transformed
# vector ihat to v[0]*transformed vector jhat


# Creates axes of plot referenced 'ax'
ax = plt.axes()

# Plots red dot at origin (0,0)
ax.plot(0,0,'or')


# Plots vector v_ihat_t as dotted green arrow starting at origin 0,0
ax.arrow(0, 0, *v_ihat_t, color='g', linestyle='dotted', linewidth=2.5, head_width=0.30,
         head_length=0.35)

# Plots vector v_jhat_t as dotted red arrow starting at origin defined by v_ihat
ax.arrow(v_ihat_t[0], v_ihat_t[1], *v_jhat_t, color='r', linestyle='dotted', linewidth=2.5,
         head_width=0.30, head_length=0.35)

# Plots vector v as blue arrow starting at origin 0,0
ax.arrow(0, 0, *v, color='skyblue', linewidth=2.5, head_width=0.30, head_length=0.35)

# DONE 4.: Plot transformed vector v (v_t) a blue colored vector(color='b') using
# vector v's ax.arrow() statement above as template for the plot
ax.arrow(0, 0, *v_t, color='b', linewidth=2.5, head_width=0.30, head_length=0.35)


# Sets limit for plot for x-axis
plt.xlim(-4, 2)

# Set major ticks for x-axis
major_xticks = np.arange(-4, 2)
ax.set_xticks(major_xticks)


# Sets limit for plot for y-axis
plt.ylim(-2, 4)

# Set major ticks for y-axis
major_yticks = np.arange(-2, 4)
ax.set_yticks(major_yticks)

# Creates gridlines for only major tick marks
plt.grid(visible=True, which='major')

# Displays final plot
plt.show()