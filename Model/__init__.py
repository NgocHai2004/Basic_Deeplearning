import torch
from torch import nn

class SimpleNeuralNetwork(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes:int = num_classes
        self.flatten = nn.Flatten()
        self.fc = nn.Sequential(
            nn.Linear(in_features=3 * 32 * 32, out_features=256),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=1024),
            nn.ReLU(),
            nn.Linear(in_features=1024, out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features = self.num_classes),
            nn.ReLU())

    def forward(self, x):
        x = self.flatten(x)
        x = self.fc(x)
        return x

