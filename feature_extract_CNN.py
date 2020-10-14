###################
# 1.2.2 Feature Extraction - Research
########################################

import os
import sys
import torch
import numpy as np
import torch.nn as nn
from pydub import AudioSegment
from torch.utils.data import Dataset
from torchsummary import summary
from sklearn.preprocessing import LabelEncoder


folderNam = 'dataset_clips'
batch_size = 5
learning_rate = 2e-4
reg = 0.01
num_epochs = 10


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
        audio = np.array((AudioSegment.from_file((os.path.join(self.audio_list[idx])), 'wav')))
        label = self.label_dict[os.path.dirname(self.audio_list[idx]).split('\\')[1][:-4]]
        return audio, label


class CNN_features(nn.Module):
    def __init__(self):
        super(CNN_features, self).__init__()

        self.cnn_layers = nn.Sequential(
            # Defining a 2D convolution layer
            nn.Conv2d(1, 4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Defining another 2D convolution layer
            nn.Conv2d(4, 4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.linear_layers = nn.Sequential(
            nn.Linear(4 * 7 * 7, 10)
        )

    # Defining the forward pass
    def forward(self, x):
        x = self.cnn_layers(x)
        x = x.view(x.size(0), -1)
        x = self.linear_layers(x)
        return x

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
print(total_step)
for epoch in range(num_epochs):
    print(epoch)
    for i, data in enumerate(train_loader):
        print('1')
        (audio, labels) = data
        print(labels)
#         # print(labels)
#         sys.exit(0)
#
#         # Forward pass
#         outputs = model(audio)
#
#         loss = criterion(outputs, labels)
#
#         # Backward and optimize
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#
#         if (i+1) % 100 == 0:
#             print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
#                    .format(epoch+1, num_epochs, i+1, total_step, loss.item()))
#
#     # Code to update the lr
#     lr *= learning_rate_decay
#     update_lr(optimizer, lr)
#     model.eval()
#     with torch.no_grad():
#         correct = 0
#         total = 0
#         for images, labels in val_loader:
#             images = images.to(device)
#             labels = labels.to(device)
#             outputs = model(images)
#             _, predicted = torch.max(outputs.data, 1)
#             total += labels.size(0)
#             correct += (predicted == labels).sum().item()
#
#         print('Validataion accuracy is: {} %'.format(100 * correct / total))
#         #################################################################################
#         # TODO: Q2.b Implement the early stopping mechanism to save the model which has #
#         # acheieved the best validation accuracy so-far.                                #
#         #################################################################################
#         best_model = None
#         # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
#
#         # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
#
#     model.train()
#
#
#
# #
# # # Iterate through all the files in the folder
# # for dirpath, dirnames, files in os.walk(folderNam):
# #     print(f'Found directory: {dirpath}')
# #
# #     for file_name in files:
# #         music_file = os.path.join(dirpath, file_name)  # path + filename
# #         print(music_file)
# #         myaudio = AudioSegment.from_file(music_file, "wav")
# #         y_pred = model(myaudio)
