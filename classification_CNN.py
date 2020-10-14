


from data_processor import *
import os
import sys
import torch
import numpy as np
import torch.nn as nn
from pydub import AudioSegment
from torch.utils.data import Dataset
from torchsummary import summary
from sklearn.preprocessing import LabelEncoder
from scipy.io.wavfile import read


folderNam = 'dataset_clips'
batch_size = 5
learning_rate = 0.1
reg = 0.1
num_epochs = 10



class CNN_features(nn.Module):
    def __init__(self):
        super(CNN_features, self).__init__()

        self.cnn_layers = nn.Sequential(
            # Defining a 2D convolution layer
            nn.Conv1d(1, 10, kernel_size=3),
            # nn.BatchNorm1d(10),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),
            # Defining another 2D convolution layer
            nn.Conv1d(10, 20, kernel_size=3),
            # nn.BatchNorm1d(20),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2)
            # Defining another 2D convolution layer
            # nn.Conv1d(20, 30, kernel_size=3),
            # nn.BatchNorm1d(4),
            # nn.ReLU(inplace=True)
            # nn.MaxPool1d(kernel_size=2, stride=2)
        )

        self.linear_layers = nn.Sequential(
            nn.Linear(20*5, 4),
            nn.Softmax(dim=1)
        )

    # Defining the forward pass
    def forward(self, x):
        x = self.cnn_layers(x)
        x = x.view(x.size(0), -1)
        x = self.linear_layers(x)
        return x


model = CNN_features()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=reg)

# Load data
X_train, X_test, y_train, y_test = data_process()
# np.random.shuffle([X_train, y_train])
X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
X_train = torch.from_numpy(X_train).float()
y_train = torch.from_numpy(y_train).long()
# Train the model
lr = learning_rate
total_step = len(X_train)
for epoch in range(num_epochs):
    print(summary(model, X_train))
    # Forward pass
    outputs = model(X_train)
    loss = criterion(outputs, y_train)

    # Backward and optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# Test the model
with torch.no_grad():
    total = np.size(y_test)
    print(total)
    y_test = torch.from_numpy(y_test).float()
    X_test = torch.from_numpy(X_test).float()
    X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])

    outputs = model(X_test)
    _, predicted = torch.max(outputs.data, 1)
    correct = int(sum(predicted == y_test))
    print(correct)

    print('Accuracy ', (100 * correct / total))