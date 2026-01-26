import torch
import torch.nn as nn

class Conv2(nn.Module):
    def __init__(self):
        super(Conv2, self).__init__()
        # Conv layers: 64, 64 filters
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        # Fully connected layers: 256, 256, 10
        self.classifier = nn.Sequential(
            nn.Linear(64 * 16 * 16, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 10)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)