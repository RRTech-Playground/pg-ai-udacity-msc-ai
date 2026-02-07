import torch
from torchvision import datasets, transforms

# Define a transform to normalize the data
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,)),
                                ])

# Download and load the training data
trainset = datasets.MNIST('~/.pytorch/MNIST_data/', download=True, train=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

# Creating the interator
dataiter = iter(trainloader)
images, labels = next(dataiter)

# Activation function
def activation(x):
    return 1/(1+torch.exp(-x))

# Flattening the images
inputs = images.view(images.shape[0], -1)
#inputs = images.view(images.shape[0], 784) # would work as well, but -1 gives us the 784 without that we know it actually

# Create parameters
w1 = torch.randn(784, 256)
b1 = torch.randn(256)
w2 = torch.randn(256, 10)
b2 = torch.randn(10)

h = activation(torch.mm(inputs, w1) + b1)

out = torch.mm(h, w2) + b2

#####
# **Exercise:**
# That's not that simple! Whatch the video for more explanations.

# Implement a functin softmax that performs the softmax calculation and returns probability distributions for each example in the batch.
# Note that you'll need to pay attention to the shapes when doing this. If you have a tensor a with shape (64, 10) and a
# tensor b with shape (64,), doing a/b will give you an error because PyTorch will try to do the division across the columns (called broadcasting)
# but you'll get a size mismatch. The way to think about this is for each of the 64 examples, you only want to divide by one value,
# the sum in the denominator. So you need b to have a shape of (64, 1). This way PyTorch will divide the 10 values in each row of
# a by the one value in each row of b. Pay attention to how you take the sum as well. You'll need to define the dim keyword in torch.sum.
# Setting dim=0 takes the sum across the rows while dim=1 takes the sum across the columns.

# Softmax
def softmax(x):
    return torch.exp(x) / torch.sum(torch.exp(x), dim=1).view(-1, 1)

probabilities = softmax(out)

# Does it have the right shape? Should be (64, 10)
print(probabilities.shape)

# Does it sum to 1?
print(probabilities.sum(dim=1))