import torch
import torch.nn as nn
import numpy as np


class FCDQN(nn.Module):

    def __init__(self, n_states, n_actions):
        super().__init__()
        self.n_states = n_states
        self.n_actions = n_actions
        self.fc1 = nn.Linear(n_states, 50)
        self.fc1.weight.data.normal_(0, 0.1)   # initialization
        self.out = nn.Linear(50, n_actions)
        self.out.weight.data.normal_(0, 0.1)   # initialization
    
    def forward(self, x):
        x = self.fc1(x)
        x = torch.relu(x)
        actions_value = self.out(x)
        return actions_value


class DQN(nn.Module):

    def __init__(self, in_channels, num_actions):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)

        self.fc1 = nn.Linear(in_features=6*9*64, out_features=512)
        self.fc2 = nn.Linear(in_features=512, out_features=num_actions)

        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class PolicyNet(DQN):

    def __init__(self, in_channels, num_actions):
        super().__init__(in_channels, num_actions)
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, x):
        x = super().forward(x)
        x = self.softmax(x)
        return x
