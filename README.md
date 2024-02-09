# Mushroom Classification Neural Network

This repository contains the implementation of a simple neural network model designed to classify mushrooms into either edible or poisonous categories based on their characteristics. The model is built using PyTorch, a popular deep learning library.

## Project Overview

The project uses a dataset from the UCI Machine Learning Repository, specifically the Mushroom Data Set, to train a neural network for the binary classification task. The model aims to accurately predict whether a mushroom is edible or poisonous based on various features such as cap shape, cap color, gill size, and others.

## Installation

Before running the project, ensure you have Python 3.x installed along with the following Python libraries:
- PyTorch
- Pandas
- NumPy
- Matplotlib
- scikit-learn

You can install the necessary libraries using pip:

```bash
pip install torch pandas numpy matplotlib scikit-learn

## Usage 
To run the neural network model, simply execute the main script:
```
python mushroom_classifier.py
```

The script will perform the following steps:

Load and preprocess the dataset.
Split the dataset into training and testing sets.
Define and train the neural network model.
Evaluate the model's performance on the test set.
Display a classification report and confusion matrix.
Model Architecture
The neural network consists of an input layer, one hidden layer, and an output layer. The ReLU activation function is used in the hidden layer, and a LogSoftmax function is used in the output layer with Negative Log-Likelihood Loss (NLLLoss) as the loss function for training.

