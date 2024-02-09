%matplotlib inline

# Import necessary libraries
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

# Custom module to fetch dataset from UCI repository
from ucimlrepo import fetch_ucirepo 

# Fetch mushroom dataset from UCI repository
mushroom = fetch_ucirepo(id=73) 

# Display dataset metadata and variable information
print(mushroom.metadata) 
print(mushroom.variables) 

# Data preprocessing
X = mushroom.data.features 
y = mushroom.data.targets 
print(X.dtypes)

# Convert categorical variables into dummy/indicator variables
X_encoded = pd.get_dummies(X)
y_encoded = y['poisonous'].map({'e': 0, 'p': 1})  # Encoding target variable

# Convert to PyTorch tensors
X_tensor = torch.FloatTensor(X_encoded.values)
y_tensor = torch.LongTensor(y_encoded.values)

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_tensor, y_tensor, test_size=0.2, random_state=100)

# Create data loaders for batching
train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)
train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)

# Define the neural network architecture
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),  # Activation function
            nn.Linear(hidden_size, num_classes),
            nn.LogSoftmax(dim=1)  # For multi-class classification
        )        
    
    def forward(self, x):
        return self.layers(x)

# Model parameters
input_size = X_train.shape[1]
hidden_size = 100  # Number of neurons in the hidden layer
num_classes = 2  # For binary classification (edible or poisonous)

# Initialize the model
model = NeuralNet(input_size, hidden_size, num_classes)

# Loss function and optimizer
criterion = nn.NLLLoss()  # Suitable for classification with LogSoftmax
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training the model
num_epochs = 20
losses = []  # To record the loss per epoch

for epoch in range(num_epochs):
    epoch_loss = 0.0
    for inputs, labels in train_loader:
        optimizer.zero_grad()  # Zero the parameter gradients
        outputs = model(inputs)  # Forward pass
        loss = criterion(outputs, labels)  # Compute loss
        loss.backward()  # Backward pass
        optimizer.step()  # Optimize

        epoch_loss += loss.item()

    epoch_loss /= len(train_loader)  # Average loss
    losses.append(epoch_loss)
    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss}')

# Evaluate the model
model.eval()  # Set the model to evaluation mode
y_pred = []
y_true = []

with torch.no_grad():
    for inputs, labels in test_loader:
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        y_pred.extend(predicted.numpy())
        y_true.extend(labels.numpy())

# Display classification report and confusion matrix
print("Classification Report:")
print(classification_report(y_true, y_pred))

cm = confusion_matrix(y_true, y_pred)
print("Confusion Matrix:")
print(cm)

# Plot the confusion matrix
plt.figure(figsize=(10,7))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()
tick_marks = np.arange(num_classes)
plt.xticks(tick_marks, range(num_classes))
plt.yticks(tick_marks, range(num_classes))
plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.show()
