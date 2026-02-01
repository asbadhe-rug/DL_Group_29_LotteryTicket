import torch
import torch.nn as nn
from tqdm import tqdm

from pruning import apply_mask

def train(model, train_loader, optimizer, criterion, device, mask):
    """
    trains the model for one epoch and returns average loss
    """
    model.train() #set model to training mode
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad() #resets gradient after each batch
        outputs = model(inputs) #passes input through model
        loss = criterion(outputs, labels) #calculates loss
        loss.backward() #computes gradient of loss 

        #set gradients of prune weights to 0
        for name, param in model.named_parameters():
            if name in mask:
                param.grad.data.mul_(mask[name])
        
        optimizer.step() #updates weights
        
        #re-apply mask to ensure pruned weights are still 0
        apply_mask(model, mask)
        
        running_loss += loss.item()
    return running_loss / len(train_loader)


def evaluate(model, loader, criterion, device, mask=None):
    model.eval() #set model to evaluation mode
    correct = 0
    total = 0
    loss = 0.0
    with torch.no_grad(): #turns off gradient tracking because no need \ (•◡•) /
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs) #passes input rtough model 
            loss += criterion(outputs, labels).item() #update total loss
            _, predicted = outputs.max(1) #give maximum value label
            total += labels.size(0) #adds number of samples 
            correct += predicted.eq(labels).sum().item() #amount of correct predictions
            
    accuracy = 100. * correct / total 
    return loss / len(loader), accuracy