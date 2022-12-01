import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear, ReLU, Sequential, Conv2d, MaxPool2d, BatchNorm2d, Dropout

class fc_model(nn.Module):

    '''
    Our fully-Connected Network for classification
    '''

    def __init__(self):
        super(fc_model, self).__init__()

        self.in_features = 224
        self.out_dim = 3

        self.fc1 = nn.Linear(self.in_features, 2048)
        self.fc2 = nn.Linear(2048, 512)
        self.fc3 = nn.Linear(512, 64)
        self.fc4 = nn.Linear(64, self.out_dim)
        self.drop1 = nn.Dropout2d(0.15)
        self.drop2 = nn.Dropout2d(0.25)
        self.drop3 = nn.Dropout2d(0.5)
        '''
        self.cnn_layers = Sequential(
            # Defining a 2D convolution layer
            Conv2d(1, 4, kernel_size=3, stride=1, padding=1),
            BatchNorm2d(4),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=2, stride=2),
            # Defining another 2D convolution layer
            Conv2d(4, 4, kernel_size=3, stride=1, padding=1),
            BatchNorm2d(4),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=2, stride=2),
            # Defining another 2D convolution layer
            Conv2d(4, 4, kernel_size=3, stride=1, padding=1),
            BatchNorm2d(4),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=2, stride=2),
        )
        '''

    def forward(self, x):

        x = self.fc1(x)
        x = F.relu(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.drop2(x)
        x = self.fc3(x)
        x = F.relu(x)
        x = self.drop(x)
        x = self.fc4(x)

        # Applying softmax

        out = F.log_softmax(x, dim=1)
        return out
