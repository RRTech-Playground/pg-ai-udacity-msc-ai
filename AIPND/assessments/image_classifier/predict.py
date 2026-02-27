import argparse
import torch
from torch import nn
from torchvision import models
from PIL import Image
import numpy as np
import json
from collections import OrderedDict

def get_input_args():
    """
    Parses command line arguments.
    """
    parser = argparse.ArgumentParser(description='Predict flower name from an image along with the probability of that name.')
    
    # Positional mandatory arguments
    parser.add_argument('image_path', type=str, help='Path to the image')
    parser.add_argument('checkpoint', type=str, help='Path to the checkpoint')
    
    # Optional arguments
    parser.add_argument('--top_k', type=int, default=1, help='Return top K most likely classes')
    parser.add_argument('--category_names', type=str, help='Path to a JSON file mapping categories to real names')
    parser.add_argument('--gpu', action='store_true', help='Use GPU for inference if available')
    
    return parser.parse_args()

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
        # Generic handling for other architectures
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

def load_checkpoint(filepath):
    """
    Loads a checkpoint and rebuilds the model.
    """
    # Load on CPU first to avoid issues if GPU is not available
    checkpoint = torch.load(filepath, map_location=lambda storage, loc: storage, weights_only=False)
    
    # Handle different possible keys for metadata
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

def process_image(image_path):
    """
    Scales, crops, and normalizes a PIL image for a PyTorch model,
    returns a Numpy array.
    """
    img = Image.open(image_path)
    
    # Resize where shortest side is 256
    if img.size[0] > img.size[1]:
        img.thumbnail((10000, 256))
    else:
        img.thumbnail((256, 10000))
        
    # Crop center 224x224
    width, height = img.size
    left = (width - 224) / 2
    top = (height - 224) / 2
    right = (width + 224) / 2
    bottom = (height + 224) / 2
    img = img.crop((left, top, right, bottom))
    
    # Normalize to [0, 1]
    np_image = np.array(img) / 255.0
    
    # Normalize with ImageNet mean and std
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    np_image = (np_image - mean) / std
    
    # Transpose to (Channels, Height, Width) as expected by PyTorch
    np_image = np_image.transpose((2, 0, 1))
    
    return np_image

def predict(image_path, model, topk, device):
    """
    Predict the class (or classes) of an image using a trained deep learning model.
    """
    model.to(device)
    model.eval()
    
    # Process image
    img = process_image(image_path)
    img_tensor = torch.from_numpy(img).type(torch.FloatTensor)
    img_tensor = img_tensor.unsqueeze_(0) # Add batch dimension
    img_tensor = img_tensor.to(device)
    
    with torch.no_grad():
        output = model.forward(img_tensor)
        
    ps = torch.exp(output)
    top_p, top_class = ps.topk(topk, dim=1)
    
    # Convert to lists
    top_p = top_p.cpu().numpy().tolist()[0]
    top_class = top_class.cpu().numpy().tolist()[0]
    
    # Invert class_to_idx to get class labels from indices
    idx_to_class = {v: k for k, v in model.class_to_idx.items()}
    top_labels = [idx_to_class[c] for c in top_class]
    
    return top_p, top_labels

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
    
    # Load model from checkpoint
    print(f"Loading model from {args.checkpoint}...")
    model = load_checkpoint(args.checkpoint)
    
    # Perform prediction
    print(f"Predicting classes for {args.image_path}...")
    top_p, top_labels = predict(args.image_path, model, args.top_k, device)
    
    # Map labels to category names if mapping file is provided
    if args.category_names:
        with open(args.category_names, 'r') as f:
            cat_to_name = json.load(f)
        top_names = [cat_to_name.get(label, f"Unknown ({label})") for label in top_labels]
    else:
        top_names = top_labels
        
    # Print results
    print("\nResults:")
    for i in range(len(top_names)):
        print(f"Rank {i+1}: {top_names[i]:<20} Probability: {top_p[i]:.4f}")

if __name__ == "__main__":
    main()