import torch
import torch.nn as nn
import torch.nn.functional as F


class QNetworkFullyConnected(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_shape, action_size, seed, fc1_units=64, fc2_units=64):
        """Initialize parameters and build model.
        Params
        ======
            state_shape (tuple): tuple with one element, state size
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super().__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_shape[0], fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)

    def forward(self, state):
        """Build a network that maps state -> action values."""
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class QNetworkFullyConvolutional(nn.Module):

    """Actor (Policy) Model."""

    def __init__(self, state_shape, action_size, seed):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        super().__init__()
        self.seed = torch.manual_seed(seed)
        _, _, _, in_channels = state_shape
        self.conv1 = nn.Conv2d(in_channels, 16, 3)
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2))
        self.drop_out = nn.Dropout()
        self.fc1 = nn.Linear(18 * 18 * 32, 32)
        self.fc2 = nn.Linear(32, action_size)

    def forward(self, state):
        """Build a network that maps state -> action values."""
        out = self.layer1(state)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.drop_out(out)
        out = F.relu(self.fc1(out))
        out = self.fc2(out)
        return out
