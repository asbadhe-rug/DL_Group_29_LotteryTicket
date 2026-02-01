import torch
import torch.nn as nn
from tqdm import tqdm

from pruning import apply_mask

def train(model, train_loader, optimizer, criterion, device, mask):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()

        for name, param in model.named_parameters():
            if name in mask and param.grad is not None:
                param.grad.data.mul_(mask[name])
        
        optimizer.step()
        apply_mask(model, mask)
        
        running_loss += loss.item()
    
    # Returns average training loss for this epoch
    return running_loss / len(train_loader)

def evaluate(model, loader, criterion, device):
    model.eval()
    correct = 0
    total = 0
    running_loss = 0.0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
    accuracy = 100. * correct / total
    # Returns average val loss and accuracy
    return running_loss / len(loader), accuracy