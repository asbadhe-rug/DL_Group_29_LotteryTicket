import torch
import torch.nn as nn

class Conv2(nn.Module):
    def __init__(self, use_dropout=False):
        super(Conv2, self).__init__()
        # Architecture: [64, 64, M]
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2, 2) # 32x32 -> 16x16
        )
        
        # Classifier layers
        layers = [
            nn.Linear(64 * 16 * 16, 256), nn.ReLU()
        ]
        if use_dropout: layers.append(nn.Dropout(0.5))
        
        layers.extend([
            nn.Linear(256, 256), nn.ReLU()
        ])
        if use_dropout: layers.append(nn.Dropout(0.5))
        
        layers.append(nn.Linear(256, 10))
        
        self.classifier = nn.Sequential(*layers)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


class Conv4(nn.Module):
    def __init__(self, use_dropout=False):
        super(Conv4, self).__init__()
        # Architecture: [64, 64, M] -> [128, 128, M]
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2, 2), # 32x32 -> 16x16
            
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2, 2)  # 16x16 -> 8x8
        )
        
        layers = [
            nn.Linear(128 * 8 * 8, 256), nn.ReLU()
        ]
        if use_dropout: layers.append(nn.Dropout(0.5))
        
        layers.extend([
            nn.Linear(256, 256), nn.ReLU()
        ])
        if use_dropout: layers.append(nn.Dropout(0.5))
        
        layers.append(nn.Linear(256, 10))
        
        self.classifier = nn.Sequential(*layers)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


class Conv6(nn.Module):
    def __init__(self, use_dropout=False):
        super(Conv6, self).__init__()
        # Architecture: [64, 64, M] -> [128, 128, M] -> [256, 256, M]
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2, 2), # 32x32 -> 16x16
            
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2, 2), # 16x16 -> 8x8
            
            nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2, 2)  # 8x8 -> 4x4
        )
        
        layers = [
            nn.Linear(256 * 4 * 4, 256), nn.ReLU()
        ]
        if use_dropout: layers.append(nn.Dropout(0.5))
        
        layers.extend([
            nn.Linear(256, 256), nn.ReLU()
        ])
        if use_dropout: layers.append(nn.Dropout(0.5))
        
        layers.append(nn.Linear(256, 10))
        
        self.classifier = nn.Sequential(*layers)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)