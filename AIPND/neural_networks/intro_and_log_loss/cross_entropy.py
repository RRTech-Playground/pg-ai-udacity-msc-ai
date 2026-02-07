import numpy as np
# TODO: complete the function
def cross_entropy(Y, P):
    """
    This function takes as input two lists Y and P, and returns the float
    corresponding to their cross-entropy.
    """

    Y = np.float_(Y)
    P = np.float_(P)

    result = -np.sum(Y * np.log(P) + (1 - Y) * np.log(1 - P))

    return result

