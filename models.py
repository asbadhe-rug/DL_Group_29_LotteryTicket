import torch
import torch.nn as nn

#creates a 2D convolutional model
class Conv2(nn.Module):
    def __init__(self, with_dropout=False):
        super(Conv2, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(), 
            nn.MaxPool2d(2, 2)
        )
        
        # Classifier layers
        layers = [nn.Linear(64 * 16 * 16, 256), nn.ReLU()] 
        if with_dropout: layers.append(nn.Dropout(0.5)) #add dropout 
        
        layers.extend([nn.Linear(256, 256), nn.ReLU()]) 
        if with_dropout: layers.append(nn.Dropout(0.5)) #add dropout
        
        layers.append(nn.Linear(256, 10)) #add final linear layer with input 256 and output 10, as CIFAR10 has 10 classes.
        
        self.classifier = nn.Sequential(*layers) #combine all layers into sequential module 


    def forward(self, x):
        """ 
        runs the input through our model.
        """
        x = self.features(x) #runs the input through the convolutional layers 
        x = x.view(x.size(0), -1) #flattens the input
        return self.classifier(x) #runs the input through the linear layers

#creates a 2D convolutial model like Conv2 but with extra convolutional layers 
class Conv4(nn.Module):
    def __init__(self, with_dropout=False):
        super(Conv4, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2, 2), # 32x32 -> 16x16
            
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2, 2)  
        )
        
        layers = [nn.Linear(128 * 8 * 8, 256), nn.ReLU()]
        if with_dropout: layers.append(nn.Dropout(0.5))
        
        layers.extend([nn.Linear(256, 256), nn.ReLU()])
        if with_dropout: layers.append(nn.Dropout(0.5))
        
        layers.append(nn.Linear(256, 10))
        
        self.classifier = nn.Sequential(*layers)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

#creates a 2D convolutial model like Conv4 but with extra convolutional layers
class Conv6(nn.Module):
    def __init__(self, with_dropout=False):
        super(Conv6, self).__init__()
    
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2, 2), 
            
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2, 2), 
            
            nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2, 2)  
        )
        
        layers = [nn.Linear(256 * 4 * 4, 256), nn.ReLU()]
        if with_dropout: layers.append(nn.Dropout(0.5))
        
        layers.extend([nn.Linear(256, 256), nn.ReLU()])
        if with_dropout: layers.append(nn.Dropout(0.5))
        
        layers.append(nn.Linear(256, 10))
        
        self.classifier = nn.Sequential(*layers)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)