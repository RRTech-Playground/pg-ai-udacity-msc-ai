import matplotlib.pyplot as plt

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models

from collections import OrderedDict

import helper
from AIPND.neural_networks.pytorch.part1_tensors_1_mm import weights

data_dir = 'data/Cat_Dog_data'

# Most of the pretrained models require the input to be 224x224 images.
# Also, we'll need to match the normalization used when the models were trained. Each color channel was normalized separately,
# the means are [0.485, 0.456, 0.406] and the standard deviations are [0.229, 0.224, 0.225].

# Define transforms for the training data and testing data
train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])

test_transforms = transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])

# Pass transforms in here,
train_data = datasets.ImageFolder(data_dir + '/train', transform=train_transforms)
test_data = datasets.ImageFolder(data_dir + '/test', transform=test_transforms)

trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
testloader = torch.utils.data.DataLoader(test_data, batch_size=64)

# then print the model to see how the transforms look
model = models.densenet121(weights=models.ResNet50_Weights.DEFAULT)
#print(model)

# Freeze parameters so we don't backprop through them
for param in model.parameters():
    param.requires_grad = False

classifier = nn.Sequential(OrderedDict([
    ('fc1', nn.Linear(1024, 500)),
    ('relu', nn.ReLU()),
    ('fc2', nn.Linear(500, 2)), # We have 2 outputs, cats and dogs
    ('output', nn.LogSoftmax(dim=1))
]))

model.classifier = classifier # Replace the model's classifier with the new classifier', the new classifier is not trained yet

print("ringgi")