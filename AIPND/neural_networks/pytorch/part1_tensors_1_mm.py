import torch

def activation(x):
    """ Sigmoid activation function

        Arguments
        ---------
        x: torch.Tensor
    """
    return 1/(1+torch.exp(-x))

### Generate some data
torch.manual_seed(7) # Set the random seed so things are predictable

# Features are 3 random normal variables
features = torch.randn((1,5))
# True weights for our data, random normal variables again
weights = torch.randn_like(features)
# and a true bias term
bias = torch.randn((1, 1))

## Calculate the output of this network using the weights and bias tensors
#y = activation(torch.sum(features * weights) + bias)
#y = activation((features * weights).sum() + bias)

#print(features.shape)
#print(weights.shape)

## Calculate the output of this network using matrix multiplication
y = activation(torch.mm(features, weights.view(5, 1)) + bias)
print(y)
