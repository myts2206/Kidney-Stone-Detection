# Kidney Stone Classifier

## Overview
This project implements a deep learning-based classifier using **VGG16**, **InceptionV3**, and **MobileNetV2** to classify kidney ultrasound images into two categories: `Normal` and `Stone`. The model is trained using PyTorch and evaluated with standard classification metrics.

## Features
- Implements three deep learning models: **VGG16, InceptionV3, and MobileNetV2**
- Uses **transfer learning** with pre-trained weights
- **Data augmentation** to improve model generalization
- **Evaluation metrics** include accuracy, precision, recall, F1-score, confusion matrix, and ROC curves
- Supports **GPU acceleration (CUDA)** for faster training

## Installation
### Prerequisites
Ensure you have Python and the necessary dependencies installed:
```bash
pip install torch torchvision numpy matplotlib seaborn tqdm scikit-learn
```

## Dataset Structure
The dataset should be structured as follows:
```
kidney_dataset/
    ├── normal/
    │   ├── train/
    │   │   ├── image1.jpg
    │   │   ├── image2.jpg
    │   ├── test/
    │       ├── image1.jpg
    │       ├── image2.jpg
    ├── stone/
        ├── train/
        │   ├── image1.jpg
        │   ├── image2.jpg
        ├── test/
            ├── image1.jpg
            ├── image2.jpg
```

## Training the Model
To train the models, run:
```bash
python main.py
```
This will train and save the best models for `VGG16`, `InceptionV3`, and `MobileNetV2` as `.pth` files.

## Evaluating the Model
To evaluate a trained model, use:
```bash
python evaluate.py --model vgg16 --weights kidney_classifier_vgg16.pth
```
Replace `vgg16` with `inception` or `mobilenet` as needed.

## Model Saving and Loading
If you encounter `_pickle.UnpicklingError`, ensure that models are saved and loaded correctly:
```python
# Save model
torch.save(model.state_dict(), "kidney_classifier_vgg16.pth")

# Load model
model.load_state_dict(torch.load("kidney_classifier_vgg16.pth", map_location=device))
```

## Results
After evaluation, the following metrics are displayed:
- **Accuracy**
- **Precision & Recall**
- **F1 Score**
- **Confusion Matrix**
- **ROC Curve**



