###################
# 1.2.2 Feature Extraction - Research
########################################

import os
import sys
import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import Dataset
from torchsummary import summary
from scipy.io.wavfile import read

folderNam = 'dataset_clips'
batch_size = 100
learning_rate = 2e-4
reg = 0.01
num_epochs = 10


# Data loader class
class AudioLoader(Dataset):
    def __init__(self, dataset_path='dataset_clips'):
        self.audio_list = []
        self.label_dict = {'Dark_Forest':0,
                           'Full-On'    :1,
                           'Goa'        :2,
                           'Hi_Tech'    :3}

        for dirpath, dirnames, files in os.walk(dataset_path):
            print(f'Found directory: {dirpath}')
            for file_name in files:
                self.audio_list.append(os.path.join(dirpath, file_name))

    def __len__(self):
        return len(self.audio_list)

    def __getitem__(self, idx):
        # print(self.audio_list[idx])
        audio = read((self.audio_list[idx]))
        # print(audio[1])
        # print(np.size(audio[1]))
        label = self.label_dict[os.path.dirname(self.audio_list[idx]).split('\\')[1][:-4]]
        return audio[1], label


# Model definition
class CNN_features(nn.Module):
    def __init__(self):
        super(CNN_features, self).__init__()

        self.cnn_layers = nn.Sequential(
            # Defining a 2D convolution layer
            nn.Conv1d(2, 4, kernel_size=3, stride=1, dilation=1),
            # nn.BatchNorm2d(4),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=5, stride=5),
            # # Defining another 2D convolution layer
            nn.Conv1d(4, 10, kernel_size=3, stride=1, dilation=3),
            # nn.BatchNorm2d(4),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=5, stride=5),
            # # Defining another 2D convolution layer
            nn.Conv1d(10, 15, kernel_size=3, stride=1, dilation=9),
            # nn.BatchNorm2d(4),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=5, stride=5),
            nn.Conv1d(15, 20, kernel_size=3, stride=1, dilation=27),
            # nn.BatchNorm2d(4),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=5, stride=5),
            nn.Conv1d(20, 25, kernel_size=3, stride=1, dilation=81),
            # nn.BatchNorm2d(4),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(25, 30, kernel_size=3, stride=1, dilation=243),
            # nn.BatchNorm2d(4),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=5, stride=5),
        )

        self.linear_layers = nn.Sequential(
            nn.Linear(2910, 200),
            nn.ReLU(inplace=True),
            nn.Linear(200, 4),
            nn.Softmax(dim=1)
        )

    # Defining the forward pass
    def forward(self, x):

        x = self.cnn_layers(x.permute(0,2,1))
        x = x.view(x.size(0), -1)
        x = self.linear_layers(x)
        return x


# Load the data
audio_loader = AudioLoader()

train_loader = torch.utils.data.DataLoader(audio_loader,
                                           batch_size=batch_size,
                                           shuffle=True)
model = CNN_features()
print(summary(model))
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=reg)

# Train the model
lr = learning_rate
total_step = len(train_loader)
for epoch in range(num_epochs):
    correct = 0
    total = 0
    for i, data in enumerate(train_loader):
        (audio, labels) = data
        audio = audio.float()

        # Forward pass
        outputs = model(audio)

        # Loss
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # print(torch.argmax(outputs, dim=1))
        # print(labels)
        correct += int(torch.sum(torch.argmax(outputs, dim=1) == labels))
        total += int(labels.shape[0])

    print('Accuracy : ', correct* 100/total)