# Image Classifier Project

This project contains a command-line application to train a deep learning model on a dataset of images and then use that trained model to predict the class of new images. It uses a pretrained network from `torchvision.models` and applies transfer learning to classify specific categories (e.g., different species of flowers).

## Data Requirements

The application expects a dataset directory containing three subfolders:
* `train`: Images used for training the model.
* `valid`: Images used for validating the model during training.
* `test`: Images used for testing the model's final performance.

Each of these subfolders should contain further subfolders, one for each class, where the subfolder name is the class label and it contains images for that class.

## Training

The `train.py` script allows you to train a new network on a dataset and save the model as a checkpoint.

### Basic Usage:
```bash
python train.py data_directory
```

### Options:
* Set directory to save checkpoints: `python train.py data_dir --save_dir save_directory`
* Choose architecture (e.g., vgg13, vgg16, alexnet): `python train.py data_dir --arch "vgg13"`
* Set hyperparameters: `python train.py data_dir --learning_rate 0.01 --hidden_units 512 --epochs 20`
* Use GPU for training: `python train.py data_dir --gpu`

## Prediction

The `predict.py` script uses a saved checkpoint to predict the class of an image.

### Basic Usage:
```bash
python predict.py /path/to/image checkpoint
```

### Options:
* Return top K most likely classes: `python predict.py image_path checkpoint --top_k 3`
* Use a mapping of categories to real names: `python predict.py image_path checkpoint --category_names cat_to_name.json`
* Use GPU for inference: `python predict.py image_path checkpoint --gpu`

### Example:
```bash
python predict.py "data/flowers/test/1/image_06743.jpg" "checkpoints/checkpoint.pth" --top_k 3 --category_names cat_to_name.json --gpu
```
