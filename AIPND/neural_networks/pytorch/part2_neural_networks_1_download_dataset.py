import torch
from torchvision import datasets, transforms

import matplotlib.pyplot as plt

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
print(type(images))
print(images.shape)
print(labels.shape)

# One image
plt.imshow(images[0].numpy().squeeze(), cmap='Greys_r')
plt.show()
