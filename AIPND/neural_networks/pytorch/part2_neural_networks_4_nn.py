import torch.nn.functional as F
from torch import nn


class MyNetwork(nn.Module):
    def __init__(self):
        super().__init__()

        # Inputs to hidden layer linear transformation
        self.hidden = nn.Linear(784, 256)
        # Output layer, 10 units - one for each digit
        self.output = nn.Linear(256, 10)

        # Define sigmoid activation and softmax output
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # Pass the input tensor through each of our operations
        x = self.hidden(x)
        x = self.sigmoid(x)
        x = self.output(x)
        x = self.softmax(x)

        return x

# Functional
class MyFunctionalNetwork(nn.Module):
    def __init__(self):
        super().__init__()

        # Inputs to hidden layer linear transformation
        self.hidden = nn.Linear(784, 256)
        # Output layer, 10 units - one for each digit
        self.output = nn.Linear(256, 10)

    def forward(self, x):
        # Hidden layer with sigmoid activation
        x = F.sigmoid(self.hidden(x))
        # Output layer with softmax activation
        x = F.softmax(self.output(x), dim=1)

        return x

class MyReLuNetwork(nn.Module):
    def __init__(self):
        super().__init__()

        # Inputs to hidden layer linear 1 transformation
        self.fc1 = nn.Linear(784, 128)
        # Inputs to hidden layer linear 2 transformation
        self.fc1 = nn.Linear(128, 64)
        # Output layer, 10 units - one for each digit
        self.output = nn.Linear(64, 10)

    def forward(self, x):
        # Hidden layer 1 with relu activation
        x = F.relu(self.fc1(x))
        # Hidden layer 2 with relu activation
        x = F.relu(self.fc1(x))
        # Output layer with softmax activation
        x = F.softmax(self.output(x), dim=1)

        return x


#model = MyNetwork()
#model = MyFunctionalNetwork()
model = MyReLuNetwork()
print(model)


## Set own weights and biases
print(model.fc1.weight)
print(model.fc1.bias)

# Set biases to all zeros
model.fc1.bias.data.fill_(0)
print(model.fc1.bias)

# sample from random normal with standard dev = 0.01
model.fc1.weight.data.normal_(std=0.01)
print(model.fc1.weight)
