import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
from sklearn.model_selection import train_test_split
from ucimlrepo import fetch_ucirepo
import json

# Fetch mushroom dataset
print("Fetching dataset...")
mushroom = fetch_ucirepo(id=73)
X = mushroom.data.features
y = mushroom.data.targets

# Save feature information
print("Processing feature information...")
feature_info = {}
for column in X.columns:
    # Convert all values to strings before sorting
    unique_values = [str(val) for val in X[column].unique() if pd.notna(val)]
    feature_info[column] = sorted(unique_values)

with open('feature_info.json', 'w') as f:
    json.dump(feature_info, f, indent=2)
print("Feature information saved to feature_info.json")

# Data preprocessing
print("Preprocessing data...")
X_encoded = pd.get_dummies(X)
y_encoded = y['poisonous'].map({'e': 0, 'p': 1})

# Convert to PyTorch tensors
X_tensor = torch.FloatTensor(X_encoded.values)
y_tensor = torch.LongTensor(y_encoded.values)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X_tensor, y_tensor, test_size=0.2, random_state=100)

# Create data loaders
train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)

# Neural network architecture
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_classes),
            nn.LogSoftmax(dim=1)
        )
    
    def forward(self, x):
        return self.layers(x)

# Initialize model
print("Training model...")
input_size = X_train.shape[1]
hidden_size = 100
num_classes = 2
model = NeuralNet(input_size, hidden_size, num_classes)

# Training parameters
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
num_epochs = 20

# Training loop
for epoch in range(num_epochs):
    epoch_loss = 0.0
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    
    epoch_loss /= len(train_loader)
    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}')

# Save the model
print("Saving model...")
torch.save(model.state_dict(), 'model.pth')

# Save feature names
feature_names = X_encoded.columns.tolist()
with open('feature_names.txt', 'w') as f:
    for feature in feature_names:
        f.write(f"{feature}\n")

print("Training complete! Model and feature information saved.") 