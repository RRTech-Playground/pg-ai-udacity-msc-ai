import matplotlib.pyplot as plt

import torch
from torchvision import datasets, transforms

import helper

data_dir = 'data/Cat_Dog_data/train'

transform = transforms.Compose([transforms.Resize(255),
                                transforms.CenterCrop(224),
                                transforms.ToTensor()]
)

dataset = datasets.ImageFolder(data_dir, transform=transform)

dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

# load an image and display it
images, labels = next(iter(dataloader))
helper.imshow(images[0], normalize=False)
plt.show()
