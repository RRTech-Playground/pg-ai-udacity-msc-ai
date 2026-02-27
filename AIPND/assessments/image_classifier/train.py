import argparse
import torch
from torch import nn, optim
from torchvision import datasets, transforms, models
import os
from collections import OrderedDict

def get_input_args():
    """
    Parses command line arguments.
    """
    parser = argparse.ArgumentParser(description='Train a new network on a dataset')
    
    # Positional mandatory argument
    parser.add_argument('data_dir', type=str, help='Path to the dataset directory')
    
    # Optional arguments
    parser.add_argument('--save_dir', type=str, default='checkpoints', help='Directory to save checkpoints')
    parser.add_argument('--arch', type=str, default='vgg13', help='CNN model architecture (vgg13, vgg16, alexnet)')
    parser.add_argument('--learning_rate', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--hidden_units', type=int, nargs='+', default=[512], help='Number of hidden units in the classifier (can provide multiple)')
    parser.add_argument('--epochs', type=int, default=5, help='Number of training epochs')
    parser.add_argument('--gpu', action='store_true', help='Use GPU for training if available')
    
    return parser.parse_args()

def load_data(data_dir):
    """
    Loads and transforms the dataset.
    """
    train_dir = os.path.join(data_dir, 'train')
    valid_dir = os.path.join(data_dir, 'valid')
    test_dir = os.path.join(data_dir, 'test')
    
    # Define transforms
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomRotation(30),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'valid': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }
    
    # Load datasets
    image_datasets = {
        'train': datasets.ImageFolder(train_dir, transform=data_transforms['train']),
        'valid': datasets.ImageFolder(valid_dir, transform=data_transforms['valid']),
        'test': datasets.ImageFolder(test_dir, transform=data_transforms['test'])
    }
    
    # Define dataloaders
    dataloaders = {
        'train': torch.utils.data.DataLoader(image_datasets['train'], batch_size=64, shuffle=True),
        'valid': torch.utils.data.DataLoader(image_datasets['valid'], batch_size=64),
        'test': torch.utils.data.DataLoader(image_datasets['test'], batch_size=64)
    }
    
    return image_datasets, dataloaders

def build_model(arch, hidden_units, num_classes):
    """
    Builds the model with a custom classifier.
    Supports both a single integer or a list of integers for hidden_units.
    """
    if arch.startswith('vgg'):
        model = getattr(models, arch)(weights='DEFAULT')
        input_features = model.classifier[0].in_features
    elif arch.startswith('alexnet'):
        model = models.alexnet(weights='DEFAULT')
        # AlexNet's classifier[0] is Dropout, [1] is Linear
        input_features = model.classifier[1].in_features
    else:
        print(f"Architecture {arch} not explicitly handled, attempting to load from torchvision.models...")
        model = getattr(models, arch)(weights='DEFAULT')
        if hasattr(model, 'classifier'):
            if isinstance(model.classifier, nn.Sequential):
                # Try to find the first Linear layer in classifier
                for layer in model.classifier:
                    if hasattr(layer, 'in_features'):
                        input_features = layer.in_features
                        break
                else:
                    raise ValueError(f"Could not find Linear layer in classifier for architecture {arch}")
            else:
                input_features = model.classifier.in_features
        elif hasattr(model, 'fc'):
            input_features = model.fc.in_features
        else:
            raise ValueError(f"Could not determine input features for architecture {arch}")
            
    # Freeze parameters
    for param in model.parameters():
        param.requires_grad = False
        
    # Define new classifier
    if isinstance(hidden_units, (int, float)):
        hidden_layers = [int(hidden_units)]
    else:
        hidden_layers = hidden_units # assuming it's a list
        
    classifier_layers = []
    last_in = input_features
    for h in hidden_layers:
        classifier_layers.append(nn.Linear(last_in, h))
        classifier_layers.append(nn.ReLU())
        classifier_layers.append(nn.Dropout(0.2))
        last_in = h
    
    classifier_layers.append(nn.Linear(last_in, num_classes))
    classifier_layers.append(nn.LogSoftmax(dim=1))
    
    classifier = nn.Sequential(*classifier_layers)
    
    if hasattr(model, 'classifier'):
        model.classifier = classifier
    else:
        model.fc = classifier
        
    return model

def train_model(model, dataloaders, criterion, optimizer, epochs, device):
    """
    Trains the model.
    """
    print(f"Training started on {device}...")
    steps = 0
    print_every = 40
    
    for epoch in range(epochs):
        running_loss = 0
        for images, labels in dataloaders['train']:
            steps += 1
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            logps = model.forward(images)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            if steps % print_every == 0:
                valid_loss = 0
                accuracy = 0
                model.eval()
                with torch.no_grad():
                    for images, labels in dataloaders['valid']:
                        images, labels = images.to(device), labels.to(device)
                        logps = model.forward(images)
                        batch_loss = criterion(logps, labels)
                        valid_loss += batch_loss.item()
                        
                        # Calculate accuracy
                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                
                print(f"Epoch {epoch+1}/{epochs}.. "
                      f"Train loss: {running_loss/print_every:.3f}.. "
                      f"Validation loss: {valid_loss/len(dataloaders['valid']):.3f}.. "
                      f"Validation accuracy: {accuracy/len(dataloaders['valid']):.3f}")
                running_loss = 0
                model.train()
    print("Training finished.")

def test_model(model, dataloaders, device):
    """
    Tests the model's accuracy on the test dataset.
    """
    print("Testing model on test dataset...")
    accuracy = 0
    model.to(device)
    model.eval()
    
    with torch.no_grad():
        for images, labels in dataloaders['test']:
            images, labels = images.to(device), labels.to(device)
            
            logps = model.forward(images)
            ps = torch.exp(logps)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
            
    print(f"Test accuracy: {accuracy/len(dataloaders['test']):.3f}")

def load_checkpoint(filepath):
    """
    Loads a checkpoint and rebuilds the model.
    """
    checkpoint = torch.load(filepath, map_location=lambda storage, loc: storage, weights_only=False)
    
    arch = checkpoint.get('arch')
    hidden_units = checkpoint.get('hidden_units', checkpoint.get('hidden_layers'))
    
    if 'class_to_idx' in checkpoint:
        num_classes = len(checkpoint['class_to_idx'])
    else:
        num_classes = checkpoint.get('output_size')
        
    if arch is None or hidden_units is None or num_classes is None:
        raise KeyError(f"Checkpoint at {filepath} is missing mandatory keys (arch, hidden_units/layers, class_to_idx/output_size)")

    model = build_model(arch, hidden_units, num_classes)
    model.load_state_dict(checkpoint['state_dict'])
    
    if 'class_to_idx' in checkpoint:
        model.class_to_idx = checkpoint['class_to_idx']
        
    return model

def main():
    args = get_input_args()
    
    # Set device

    device = (
        torch.accelerator.current_accelerator().type
        if args.gpu and torch.accelerator.is_available()
        else "cpu"
    )

    # Legacy PyTorch
    #device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")

    print(f"Device set to: {device}")
    
    # Load data
    image_datasets, dataloaders = load_data(args.data_dir)
    
    # Build model
    num_classes = len(image_datasets['train'].classes)
    model = build_model(args.arch, args.hidden_units, num_classes)
    
    # Define criterion and optimizer
    classifier_params = model.classifier.parameters() if hasattr(model, 'classifier') else model.fc.parameters()
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(classifier_params, lr=args.learning_rate)
    
    model.to(device)
    
    # Train model
    train_model(model, dataloaders, criterion, optimizer, args.epochs, device)
    
    # Test model
    test_model(model, dataloaders, device)
    
    # Save checkpoint
    model.class_to_idx = image_datasets['train'].class_to_idx
    checkpoint = {
        'arch': args.arch,
        'hidden_units': args.hidden_units,
        'state_dict': model.state_dict(),
        'class_to_idx': model.class_to_idx,
        'optimizer_state': optimizer.state_dict(),
        'epochs': args.epochs
    }
    
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
        
    save_path = os.path.join(args.save_dir, 'checkpoint.pth')
    torch.save(checkpoint, save_path)
    print(f"Checkpoint saved to {save_path}")

if __name__ == "__main__":
    main()
