# EXE-Binary: Malware Detection using Convolutional Neural Networks

A deep learning project for detecting malicious executable files using Convolutional Neural Networks (CNN). This project classifies EXE binaries as either **Benign** or **Malicious** by converting them to images and applying image classification techniques.

## Overview

This project implements a CNN-based malware detection system that achieves **98.54% validation accuracy**. The model uses a custom architecture with four convolutional layers, batch normalization, dropout regularization, and fully connected layers for binary classification.

## Project Structure

```
exe-binary/
├── app/
│   └── main.py                 # Main api file
├── artifacts/
│   └── best_malware_model.pth  # Trained model weights
├── notebooks/
│   └── exe-binary.ipynb        # Training notebook implementation
├── plots/
│   ├── confusion_matrix.png    # Model confusion matrix visualization
│   └── training_history.png    # Training/validation loss and accuracy plots
└── README.md
```

## Features

- **Binary Classification**: Classifies EXE files as Benign or Malicious
- **Image-based Approach**: Converts executable binaries to images for CNN processing
- **Data Augmentation**: Includes rotation, flipping, affine transformations, and color jittering
- **Model Performance**: 98.54% validation accuracy with 99% precision and recall
- **Visualization**: Generates confusion matrix and training history plots

## Model Architecture

The CNN model consists of:

- **4 Convolutional Blocks**: Each with Conv2d, BatchNorm2d, ReLU, MaxPool2d, and Dropout
  - Block 1: 32 filters
  - Block 2: 64 filters
  - Block 3: 128 filters
  - Block 4: 256 filters
- **Fully Connected Layers**: 
  - 512 → 256 → 1 (with BatchNorm and Dropout)
  - Final Sigmoid activation for binary classification

## Training Details

- **Image Size**: 224×224 pixels
- **Batch Size**: 32
- **Epochs**: 50 (with early stopping)
- **Learning Rate**: 0.001 (Adam optimizer with ReduceLROnPlateau scheduler)
- **Loss Function**: Binary Cross-Entropy Loss
- **Dataset Split**: 80% training, 20% validation
- **Training Samples**: 16,683
- **Validation Samples**: 4,171

## Requirements

- Python 3.11+
- PyTorch
- torchvision
- numpy
- matplotlib
- seaborn
- scikit-learn
- tqdm
- PIL/Pillow

## Usage

### Training

The training code is available in `notebooks/exe-binary.ipynb`. Key steps:

1. Set the dataset path:
   ```python
   DATASET_PATH = r'/path/to/your/dataset'
   ```

2. The dataset should be organized as:
   ```
   dataset/
   ├── Benign/
   │   └── *.png
   └── Malicious/
       └── *.png
   ```

3. Run the training cells in the notebook

## Results

- **Best Validation Accuracy**: 98.54%
- **Precision**: 0.99 (Benign), 0.99 (Malicious)
- **Recall**: 0.99 (Benign), 0.98 (Malicious)
- **F1-Score**: 0.99 (both classes)

## License

This project is part of a graduate research project.

