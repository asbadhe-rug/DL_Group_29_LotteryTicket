import torch
import torch.optim as optim
import torch.nn as nn
import copy
from models import Conv2, Conv4, Conv6
from utils import get_loaders
from pruning import get_mask, apply_mask
from train import train, evaluate


def reinitialize_model(model):
    """Reinitializes all weights using Gaussian Glorot (paper standard)."""
    for layer in model.modules():
        if isinstance(layer, (torch.nn.Conv2d, torch.nn.Linear)):
            torch.nn.init.xavier_normal_(layer.weight)
            if layer.bias is not None:
                torch.nn.init.constant_(layer.bias, 0)

# Hyperparameters from the paper for Conv-2
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 60
LR = 0.0002
PRUNE_RATE = 0.2  # Prune 20% of remaining weights each iteration
ITERATIONS = 5    # How many times to prune and retrain

train_loader, val_loader, test_loader = get_loaders(BATCH_SIZE)

# 1. Initialize and save the "Ticket" (theta_0)
model = Conv4().to(device)
initial_state_dict = copy.deepcopy(model.state_dict())

# Current mask (initially all ones - no pruning)
mask = {name: torch.ones_like(param) for name, param in model.named_parameters() if 'weight' in name}

for round in range(ITERATIONS):
    model.load_state_dict(copy.deepcopy(initial_state_dict))
    apply_mask(model, mask)
    print(f"\n--- Pruning Round {round} ---")
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()
    
    # 2. Train the current subnetwork
    for epoch in range(1, 11): # Start with 10 epochs for testing
        loss = train(model, train_loader, optimizer, criterion, device, mask)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        print(f"Epoch {epoch}: Val Acc = {val_acc:.2f}%")

    # 3. Prune: Identify the winning ticket's next mask
    # We increase the pruning percentage each round
    current_prune_percent = 1 - (1 - PRUNE_RATE)**(round + 1)
    mask = get_mask(model, current_prune_percent)
    
    # 4. Reset: Back to theta_0 and apply mask
    model.load_state_dict(copy.deepcopy(initial_state_dict))
    apply_mask(model, mask)
    reinitialize_model(model) # Give it fresh random weights
    apply_mask(model, mask)
    
    print(f"Pruning complete. Sparsity: {current_prune_percent*100:.1f}%")