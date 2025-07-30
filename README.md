# Deep Neural Networks and Time-Series Models

## Overview
This repository implements deep learning models for image classification and sequence classification tasks. It includes from-scratch training of AlexNet and ResNet50 CNNs on the CIFAR-10 dataset with enhancements like data augmentation, early stopping with patience, and performance visualizations. Additionally, it compares recurrent models (RNN, GRU, LSTM) for a time-series-like task: classifying names by language origin using a dataset of names from various categories. Key features include training loops, loss/accuracy tracking, activation maps, t-SNE visualizations, and confusion matrices. ResNet50 outperforms AlexNet on CIFAR-10 (74.46% vs. 62.46% test accuracy), while RNN achieves the highest accuracy (65.30%) among recurrent models for name classification.

The project is implemented in Python using PyTorch for CNNs and recurrent models, with Jupyter Notebook as the primary file (`deep-neural-networks-and-time-series-models.ipynb`).

## Prerequisites
- Python 3.x
- Libraries: `torch`, `torchvision`, `matplotlib`, `numpy`, `sklearn`, `unicodedata`, `string`, `os`, `glob`, `random`
- Datasets: CIFAR-10 (auto-downloaded via code); Names dataset (assumed in `/kaggle/input/nlp-dl/data/names/*.txt` or similar path)
- Hardware: GPU recommended for training (code uses `cuda` if available)

## Installation
1. Clone the repository:
```bash
git clone https://github.com/Jayadeep-Potarlanka/deep-neural-networks-and-time-series-models.git
```
2. Install dependencies:
```bash
pip install torch torchvision matplotlib numpy scikit-learn
```
3. Ensure the names dataset is placed in the appropriate directory (e.g., `data/names/`).

## Usage
1. Open the Jupyter notebook: `deep-neural-networks-and-time-series-models.ipynb`.
2. Run sections sequentially:
- **CNN Section**: Loads CIFAR-10, defines AlexNet/ResNet50, trains with Adam optimizer, StepLR scheduler (for ResNet50), and early stopping. Visualizes loss curves, activation maps, t-SNE embeddings, and computes test metrics.
- **Recurrent Models Section**: Loads names dataset, trains RNN/GRU/LSTM for language classification, plots training losses, evaluates with confusion matrices.
3. Example outputs:
- Training progress printed per epoch (e.g., losses and accuracies).
- Visualizations generated via Matplotlib (e.g., loss plots, activation maps, t-SNE, confusion matrices).

To run training for CNNs:
```bash
alexnet_results = train_model_with_patience(alexnet, optimizer_alexnet)
resnet50_results = train_model_with_patience(resnet50, optimizer_resnet50, scheduler_resnet50)
```

For recurrent models, training loops iterate over 100,000 examples with random sampling.

## Models and Results
### CNNs on CIFAR-10
- **Data Handling**: Enhanced augmentation (random flip, crop, rotation, resize to 224x224, normalization). Split: 70% train, 10% validation, 20% test (from train set).
- **AlexNet**: Trained for 19 epochs (early stop), test loss: 1.0807, test accuracy: 62.46%.
- **ResNet50**: Trained for 15 epochs (early stop), test loss: 0.7436, test accuracy: 74.46%.
- Comparison: ResNet50 shows better generalization due to residual connections, with clearer activation maps indicating diverse feature learning.

### Recurrent Models for Name Classification
- **Task**: Classify names into 18 language categories (e.g., Vietnamese, Greek, Japanese) as a sequence modeling problem.
- **RNN**: Accuracy: 65.30%.
- **GRU**: Accuracy: 54.70%.
- **LSTM**: Accuracy: 52.40%.
- Training: 100,000 iterations with NLL loss and manual gradient updates (learning rate 0.005). Includes confusion matrices showing model confusions (e.g., between similar languages).

Visualizations include training loss curves, activation maps (for CNNs), t-SNE for feature evolution (AlexNet), and confusion matrices (for recurrent models).
