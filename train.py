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

        # --- THIS IS THE PAPER-STRICT PART ---
        # 1. Zero out gradients for pruned weights so they don't update
        for name, param in model.named_parameters():
            if name in mask:
                param.grad.data.mul_(mask[name])
        
        optimizer.step()
        
        # 2. Safety check: ensure weights are still exactly zero
        # (Numerical errors in optimizers can sometimes create tiny values)
        apply_mask(model, mask)
        
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