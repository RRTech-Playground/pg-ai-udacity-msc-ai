import helper
import matplotlib.pyplot as plt
import torch
from torch import nn
from torchvision import datasets, transforms

# Define a transform to normalize the data
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,)),
                                ])

# Download and load the training data
trainset = datasets.MNIST('~/.pytorch/MNIST_data/', download=True, train=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

# Hyperparameters for our network
input_size = 784
hidden_sizes = [128, 64]
output_size = 10

# Build a feed-forward network
model = nn.Sequential(nn.Linear(input_size, hidden_sizes[0]),
                      nn.ReLU(),
                      nn.Linear(hidden_sizes[0], hidden_sizes[1]),
                      nn.ReLU(),
                      nn.Linear(hidden_sizes[1], output_size),
                      nn.Softmax(dim=1))
print(model)
print(model[0])
print(model[0].weight)

## You can also pass in an `OrderedDict` to name the individual layers and operations, instead of using incremental
## integers. Note that dictionary keys must be unique, so _each operation must have a different name_.

#from collections import OrderedDict
#model = nn.Sequential(OrderedDict([
#    ('fc1', nn.Linear(input_size, hidden_sizes[0])),
#    ('relu1', nn.ReLU()),
#    ('fc2', nn.Linear(hidden_sizes[0], hidden_sizes[1])),
#    ('relu2', nn.ReLU()),
#    ('output', nn.Linear(hidden_sizes[1], output_size)),
#    ('softmax', nn.Softmax(dim=1))]))


# Forward pass through the network and display output
images, labels = next(iter(trainloader))
images.resize_(images.shape[0], 1, 784)
ps = model.forward(images[0,:])
helper.view_classify(images[0].view(1, 28, 28), ps)

plt.show()