import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from matplotlib import pyplot as plt
import helper

# Define a transform to normalize the data
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize(0.5, 0.5)])

# The original transform from the notebook. But if used you need to convert it back to grayscale. The MNIST dataset is already grayscale.
transform2 = transforms.Compose([
    transforms.Grayscale(num_output_channels=3), # without this line, the images will not be grayscale and the iterator fails.
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# Download and load the training data
trainset = datasets.FashionMNIST('~/.pytorch/F_MNIST_data/', download=True, train=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

# Download and load the test data
testset = datasets.FashionMNIST('~/.pytorch/F_MNIST_data/', download=True, train=False, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=True)

#image, label = next(iter(trainloader))
#helper.imshow(image[0,:])
#plt.show()

class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 10)

    def forward(self, x):
        # make sure input tensor is flattened
        x = x.view(x.shape[0], -1)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.log_softmax(self.fc4(x), dim=1)

        return x

# Create the network, define the criterion and optimiz
model = Classifier()
criterion = nn.NLLLoss()  # As we use LogSoftmax for the output, we use Negative Log Likelihood Loss as the criterion
optimizer = optim.Adam(model.parameters(), lr=0.003)  # In contrast to SDG, Adam uses momentum by default and adjusts the learning rate for each of the individual parameters.

# Train the network
epochs = 5
for e in range(epochs):
    running_loss = 0
    for images, labels in trainloader:

        # Now we don't need to flatten the image here because we did it in the network class

        # Forward pass
        logps = model(images)
        loss = criterion(logps, labels)

        # then backward pass and update weights
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    else:
        print(f"Training loss: {running_loss/len(trainloader)}")

# Test out your network!
dataiter = iter(testloader)
images, labels = next(dataiter)
img = images[1]

# Calculate the class probabilities (softmax) for img
ps = torch.exp(model(img))

# Plot the image and probabilities
helper.view_classify(img, ps, version='Fashion')

plt.show()