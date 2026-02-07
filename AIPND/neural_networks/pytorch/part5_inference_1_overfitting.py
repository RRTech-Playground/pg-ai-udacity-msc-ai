import torch
import torch.nn.functional as F
from torch import nn, optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt


## Load the data

# Define a transform to normalize the data
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5, ), (0.5, ))])
# Download and load the training data
trainset = datasets.FashionMNIST('~/.pytorch/F_MNIST_data/', download=True, train=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

# Download and load the test data
testset = datasets.FashionMNIST('~/.pytorch/F_MNIST_data/', download=True, train=False, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=True)


## Create Model and get the predictions
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

model = Classifier()

images, labels = next(iter(testloader))

# Get the class probabilities
ps = torch.exp(model(images))

# Make sure the shape is appropriate, we should get 10 class probabilities for 64 examples
print(ps.shape)


## Let's look at the top 5, highest probability classes for our first 10 test images
top_p, top_class = ps.topk(1, dim=1)
# Look at the most likely classes for the first 10 examples
print(top_class[:10,:])


## Compare against the labels

#equals = top_class == labels # Not good enough! torch.Size([64, 64]) we need torch.Size([64, 1])
equals = top_class == labels.view(*top_class.shape)

print(equals[:10,:])

#accuracy = torch.mean(equals)  # .mean() is not implemented for a ByteTensor, we'll need to convert equals to a float tensor.
accuracy = torch.mean(equals.type(torch.FloatTensor))

print(f'Accuracy: {accuracy.item()*100}%')


# Train and compare, after each training epoch we do a validation pass
model = Classifier()
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.parameters(), lr=0.003)

epochs = 30
steps = 0

train_losses, test_losses = [], []
for e in range(epochs):
    running_loss = 0
    for images, labels in trainloader:

        optimizer.zero_grad()

        log_ps = model(images)
        loss = criterion(log_ps, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    else:
        test_loss = 0
        accuracy = 0

        # Turn off gradients for validation, saves memory and computations
        with torch.no_grad():
            for images, labels in testloader:
                log_ps = model(images)
                test_loss += criterion(log_ps, labels)

                ps = torch.exp(log_ps)
                top_p, top_class = ps.topk(1, dim=1)
                equals = top_class == labels.view(*top_class.shape)
                accuracy += torch.mean(equals.type(torch.FloatTensor))

        train_losses.append(running_loss/len(trainloader))
        test_losses.append(test_loss/len(testloader))

        print(f"Epoch {e+1}/{epochs}.. ",
              f"Training loss: {running_loss/len(trainloader):.3f}.. ",
              f"Test loss: {test_loss/len(testloader):.3f}.. ",
              f"Test accuracy: {accuracy/len(testloader):.3f}")


# Plot the training and validation loss - check for overfitting
plt.plot(train_losses, label='training loss')
plt.plot(test_losses, label='validation loss')
plt.legend(frameon=False)
plt.show()

