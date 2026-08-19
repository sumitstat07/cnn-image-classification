# CNN for CIFAR-10 Image Classification

A convolutional neural network built from scratch in PyTorch to classify images from the CIFAR-10 dataset into 10 object categories.

## Overview

This project trains a CNN to recognize 10 classes of everyday objects — airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck — from 32x32 color images. It covers the full pipeline: data loading, model architecture, training, evaluation, and inference on custom images.

## Dataset

- **CIFAR-10**: 60,000 32x32 color images across 10 classes (50,000 train / 10,000 test)
- Loaded via `torchvision.datasets.CIFAR10`, normalized to `[-1, 1]`

## Model Architecture

A simple 3-block CNN:

| Block | Layers |
|---|---|
| Conv Block 1 | Conv2d(3→32, 3x3) → ReLU → MaxPool(2x2) |
| Conv Block 2 | Conv2d(32→64, 3x3) → ReLU → MaxPool(2x2) |
| Conv Block 3 | Conv2d(64→128, 3x3) → ReLU → MaxPool(2x2) |
| Fully Connected | Linear(2048→256) → ReLU → Linear(256→10) |

Trained with Adam optimizer and Cross-Entropy loss for 10 epochs.

## Results

- **Test accuracy: 74.71%**
- Per-class accuracy, a confusion matrix, and a classification report (precision/recall/F1 per class) are included in the notebook to show where the model performs well and where it struggles (e.g. classes like cat/dog tend to get confused with each other).

## Testing on Your Own Image

Beyond the test set, the notebook includes a `predict_image()` function that lets you run inference on any image file:

```python
predicted_class, confidence = predict_image("my_photo.jpg", inference_model, transform, classes, device)
```

It resizes the image to 32x32, runs it through the trained model, and displays the image with its predicted class and confidence, plus the top-3 predictions.

**Note:** the model only knows the 10 CIFAR-10 classes above — an image of anything outside those categories will still be forced into one of the 10 labels.

## How to Run

1. Clone the repo:
   ```
   git clone https://github.com/sumitstat07/cnn-image-classification.git
   cd cnn-image-classification
   ```
2. Install dependencies:
   ```
   pip install torch torchvision scikit-learn seaborn matplotlib pillow
   ```
3. Open `CNN_for_CIFAR10.ipynb` in Jupyter and run all cells. CIFAR-10 will download automatically on first run.

## Project Structure

```
cnn-image-classification/
├── CNN_for_CIFAR10.ipynb   # main notebook: build, train, evaluate, test
├── README.md
└── .gitignore
```

## Possible Improvements

- Add batch normalization and dropout to reduce overfitting
- Add data augmentation (random crop/flip) to improve generalization
- Track validation loss per epoch alongside training loss
- Try a deeper architecture or a pretrained backbone (e.g. ResNet) for comparison
