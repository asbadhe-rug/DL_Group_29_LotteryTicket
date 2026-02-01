import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, SubsetRandomSampler
import numpy as np

def get_loaders(batch_size=60):
    """
    Downloads the CIFAR10 dataset, normalizes it and splits it into training, validation, and test part.
    """
    #creating a preprocess sequence to get the data in the right shape and have it normalized
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), #These are the mean and standard deviation of the R,G,B values of the images.
    ])

    # Downloading the CIFAR10 training and test sets
    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    # Shuffeling the training set and splitting it into a train part and validation part 
    indices = list(range(len(train_dataset)))
    np.random.shuffle(indices)
    train_idx, val_idx = indices[5000:], indices[:5000]

    #randomizing the datasets and splitting it into batches
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=SubsetRandomSampler(train_idx))
    val_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=SubsetRandomSampler(val_idx))
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader