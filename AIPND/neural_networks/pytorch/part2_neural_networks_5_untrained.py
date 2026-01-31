import helper
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from torchvision import datasets, transforms
from torch import nn

class MyReLuNetwork(nn.Module):
    def __init__(self):
        super().__init__()

        # Inputs to hidden layer linear 1 transformation
        self.fc1 = nn.Linear(784, 128)
        # Inputs to hidden layer linear 2 transformation
        self.fc2 = nn.Linear(128, 64)
        # Output layer, 10 units - one for each digit
        self.output = nn.Linear(64, 10)

    def forward(self, x):
        # Hidden layer 1 with relu activation
        x = F.relu(self.fc1(x))
        # Hidden layer 2 with relu activation
        x = F.relu(self.fc2(x))
        # Output layer with softmax activation
        x = F.softmax(self.output(x), dim=1)

        return x

model = MyReLuNetwork()

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

# Resize images into a 1D vector, new shape is (batch size, color channels, image pixels)
images.resize_(64, 1, 784)
# or images.resize_(images.shape[0], 1, 784) to automatically get batch size

# Forward pass through the network
img_idx = 0
ps = model.forward(images[img_idx,:])

img = images[img_idx]
helper.view_classify(img.view(1, 28, 28), ps)

plt.show()