import torch
import torch.nn as nn
import torch.optim as optim
import copy
import pandas as pd
import os
import time
from datetime import datetime, timedelta

# Local file imports
from models import Conv2, Conv4, Conv6 
from pruning import get_mask, apply_mask
from train import train, evaluate
from utils import get_loaders

def reinitialize_model(model):
    """
    Randomly re-initializes weights for the control group.
    Matches the 'Random Re-init' dashed lines in Figure 5.
    """
    for layer in model.modules():
        if isinstance(layer, (nn.Conv2d, nn.Linear)):
            nn.init.kaiming_normal_(layer.weight)
            if layer.bias is not None:
                nn.init.constant_(layer.bias, 0)

def run_experiment(model_name, model_class, use_dropout, experiment_type, iterations=15, epochs=25):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Batch size 60 as per the paper
    train_loader, val_loader, _ = get_loaders(batch_size=60) 
    
    # 1. Setup Model and capture Initial State (Theta_0)
    model = model_class(use_dropout=use_dropout).to(device)
    initial_state_dict = copy.deepcopy(model.state_dict())
    
    # Initialize mask as all ones (100% weights remaining)
    mask = {name: torch.ones_like(param) for name, param in model.named_parameters() if 'weight' in name}
    
    results = []

    for round_idx in range(iterations):
        round_start = time.time()
        
        # 2. Reset Step: Back to theta_0 (Winning Ticket) or New Random (Re-init)
        if experiment_type == "winning_ticket":
            model.load_state_dict(copy.deepcopy(initial_state_dict))
        else:
            reinitialize_model(model)
            
        apply_mask(model, mask)
        
        # Paper LR: 0.0003 for Conv2, 0.0002 for others
        lr = 0.0003 if model_name == "Conv2" else 0.0002
        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        
        best_acc = 0.0
        early_stop_epoch = epochs
        
        # 3. Training Loop
        print(f"\n>>> Running: {model_name} | {experiment_type} | Round {round_idx}")
        
        round_results = []
        for epoch in range(1, epochs + 1):
            train_loss = train(model, train_loader, optimizer, criterion, device, mask)
            val_loss, val_acc = evaluate(model, val_loader, criterion, device)
            
            # Track early stopping (Paper uses "minimum validation loss" or "early stop epoch")
            if val_acc > 75.0 and early_stop_epoch == epochs:
                early_stop_epoch = epoch
                
            if val_acc > best_acc:
                best_acc = val_acc
            
            round_results.append({
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_acc": val_acc
            })

        # Calculate final stats for this round
        total_weights = 0
        remaining_weights = 0
        for name, param in model.named_parameters():
            if 'weight' in name:
                total_weights += param.numel()
                remaining_weights += torch.count_nonzero(param.data).item()
        
        remaining_percent = (remaining_weights / total_weights) * 100
        
        results.append({
            "model": model_name,
            "type": experiment_type,
            "round": round_idx,
            "remaining_percent": remaining_percent,
            "best_val_acc": best_acc,
            "early_stop_epoch": early_stop_epoch,
            "final_train_loss": round_results[-1]["train_loss"], # For overfitting analysis
            "final_val_loss": round_results[-1]["val_loss"]      # For overfitting analysis
        })

        # --- TIME ESTIMATION ---
        round_duration = time.time() - round_start
        rounds_left = iterations - (round_idx + 1)
        eta_seconds = rounds_left * round_duration
        eta_time = datetime.now() + timedelta(seconds=eta_seconds)

        print(f"Round {round_idx} took {round_duration/60:.1f}m. "
              f"Estimated model completion at: {eta_time.strftime('%H:%M:%S')}")

        # 4. Pruning Step: Create mask for NEXT round (iterative pruning)
        mask = get_mask(model, conv_rate=0.1, fc_rate=0.2)

    return results

if __name__ == "__main__":
    all_data = []
    
    # We will test Conv2 and Conv4. Iterations=15 gets you to ~3.5% weights.
    # Epochs=25 aligns with the paper's 20k-25k training iterations.
    configs = [
        ("Conv2", Conv2, False), 
        ("Conv4", Conv4, False),
        ("Conv2", Conv2, True)  # Includes Dropout analysis for Fig 6
    ]
    
    types = ["winning_ticket", "random_reinit"]

    for m_name, m_class, d_out in configs:
        for t in types:
            data = run_experiment(m_name, m_class, d_out, t, iterations=15, epochs=25)
            all_data.extend(data)
            
            # Save progress so you don't lose data if it crashes
            pd.DataFrame(all_data).to_csv("imp_results_backup.csv", index=False)

    # Final Save
    final_df = pd.DataFrame(all_data)
    final_df.to_csv("final_lottery_ticket_results_layer-wise_pruning.csv", index=False)
    print("\n✅ All experiments finished. Data saved to final_lottery_ticket_results_layer-wise_pruning.csv")