import numpy as np

# Define vector v
v = np.array([-1,2])

# Define 2x2 matrix ij
ij = np.array([[3, 1],[1, 2]])

# DONE 1.: Demonstrate getting v_trfm by matrix multiplication
# by using matmul function to multiply 2x2 matrix ij by vector v
# to compute the transformed vector v (v_t)
v_t = np.matmul(ij, v)

# Prints vectors v, ij, and v_t
print("\nMatrix ij:", ij, "\nVector v:", v, "\nTransformed Vector v_t:", v_t, sep="\n")