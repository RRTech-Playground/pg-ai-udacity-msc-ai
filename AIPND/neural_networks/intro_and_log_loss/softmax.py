import numpy as np
import math

# TODO: complete the function
def softmax(L):
    """
    This function takes as input a list of numbers, and returns the list
    of values given by the softmax function.
    """

    exp_L = np.exp(L)
    result = list(np.divide(exp_L, exp_L.sum()))

    return result

#def softmax2(L):
    """
    This function takes as input a list of numbers, and returns the list
    of values given by the softmax function.
    """
    exp_L = [math.e ** i for i in L]
    sum_exp_L = sum(exp_L)
    result = [i / sum_exp_L for i in exp_L]

    return result

### Notebook grading

def correct_softmax(L):
    exp_L = np.exp(L)
    return list(np.divide(exp_L, exp_L.sum()))

L_test = [5,6,7]
solution = correct_softmax(L_test)
trial = softmax(L_test)

if len(trial) != len(solution):
    print("Hmm... there must be a mistake. Trying for L={}. The \
length of the correct result is {} but the list returned by your code \
is of length {}".format(
        L_test,
        len(solution),
        len(trial)
    ))
elif np.allclose(trial, solution):
    print("Correct!")
else:
    print("Hmm... there must be a mistake. Trying for L={}. The \
correct answer is {} and your code returned {}".format(
        L_test,
        solution,
        trial
    ))