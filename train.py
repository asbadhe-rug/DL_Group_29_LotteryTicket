import torch
import torch.nn as nn
from tqdm import tqdm

def train(model, train_loader, optimizer, criterion, device, mask=None):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        
        # IMPORTANT: If we have a mask, we must zero out the gradients 
        # for pruned weights so they never update during training.
        if mask is not None:
            for name, param in model.named_parameters():
                if name in mask:
                    param.grad.data.mul_(mask[name])
        
        optimizer.step()
        running_loss += loss.item()
    
    return running_loss / len(train_loader)

def evaluate(model, loader, criterion, device, mask=None):
    model.eval()
    correct = 0
    total = 0
    loss = 0.0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss += criterion(outputs, labels).item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
    accuracy = 100. * correct / total
    return loss / len(loader), accuracy