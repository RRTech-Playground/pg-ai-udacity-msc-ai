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
print(out.shape)
