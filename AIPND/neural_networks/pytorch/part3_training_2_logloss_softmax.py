import torch
from torch import nn
import torch.nn.functional as F
from torchvision import datasets, transforms

# Exercise
# Build a model that returns the log-softmax as the output and calculate the loss using the negative log likelihood loss.
# Note that for`nn.LogSoftmax`and`F.log_softmax`you'll need to set the`dim`keyword argument appropriately.`dim=0`calculates
# softmax across the rows, so each column sums to 1, while`dim=1`calculates across the columns so each row sums to 1.
# Think about what you want the output to be and choose`dim`appropriately.

## Step 1 - Load the data

# Define a transform to normalize the data
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,)),
                                ])
# Download and load the training data
trainset = datasets.MNIST('~/.pytorch/MNIST_data/', download=True, train=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

## Step 2 - Build the network

# Build a feed-forward network
model = nn.Sequential(nn.Linear(784, 128),
                      nn.ReLU(),
                      nn.Linear(128, 64),
                      nn.ReLU(),
                      nn.Linear(64, 10),
                      nn.LogSoftmax(dim=1))

## Step 3 - Find the error

# Define the loss
criterion = nn.NLLLoss()

# Get our data
images, labels = next(iter(trainloader))
# Flatten images
images = images.view(images.shape[0], -1)

# Forward pass, get our logits
logps = model(images)
# Calculate the loss with the logits and the labels
loss = criterion(logps, labels)

print(loss)