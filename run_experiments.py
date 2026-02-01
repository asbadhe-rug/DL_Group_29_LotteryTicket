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
from pruning import get_mask_layerwise, apply_mask
from train import train, evaluate
from utils import get_loaders

def reinitialize_model(model):
    """
    Randomly re-initializes weights for the control group.
    Matches the 'Random Re-init' dashed lines in Figure 5.
    """
    for layer in model.modules():
        if isinstance(layer, (nn.Conv2d, nn.Linear)):
            nn.init.xavier_normal_(layer.weight)
            if layer.bias is not None:
                nn.init.constant_(layer.bias, 0)

def run_experiment(model_name, model_class, use_dropout, experiment_type, iterations=15, epochs=25):
    # Determine per-iteration pruning rates based on model (Figure 2 in paper)
    if model_name == "Conv2" or model_name == "Conv4":
        conv_prune_rate = 0.10  # Prune 10% of remaining conv weights per iteration
        fc_prune_rate = 0.20    # Prune 20% of remaining FC weights per iteration
    elif model_name == "Conv6":
        conv_prune_rate = 0.15  # Prune 15% of remaining conv weights per iteration
        fc_prune_rate = 0.20    # Prune 20% of remaining FC weights per iteration
    else:
        # Default fallback (shouldn't hit this with current configs)
        conv_prune_rate = 0.20
        fc_prune_rate = 0.20
    
    # Adjust epochs based on dropout
    # With dropout, networks need 3x training time
    if use_dropout:
        epochs = epochs * 3  # 75 epochs for dropout (vs 25 without)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Batch size 60 as per the paper
    train_loader, val_loader, test_loader = get_loaders(batch_size=60) 
    
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

        # Determine learning rate based on model and dropout (Appendix H.5)
        if use_dropout:
            # With dropout, use higher LR for Conv2, lower LR for Conv4/6
            if model_name == "Conv2":
                lr = 0.0003  # 3 × 10^-4
            else:  # Conv4 and Conv6
                lr = 0.0002  # 2 × 10^-4
        else:
            # Without dropout, use standard rates
            if model_name == "Conv2":
                lr = 0.0002  # 2 × 10^-4
            else:  # Conv4 and Conv6
                lr = 0.0003  # 3 × 10^-4
        
        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        
        best_val_loss = float('inf')
        early_stop_epoch = epochs
        best_state_dict = None  # Save weights at early-stop iteration
        
        # 3. Training Loop
        print(f"\n>>> Running: {model_name} | {experiment_type} | Round {round_idx} (Target: {epochs} Epochs)")
        
        for epoch in range(1, epochs + 1):
            train_loss = train(model, train_loader, optimizer, criterion, device, mask)
            val_loss, val_acc = evaluate(model, val_loader, criterion, device)
            
            # Track best validation loss for early stopping (as per paper Appendix C)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                early_stop_epoch = epoch
                best_state_dict = copy.deepcopy(model.state_dict())  # Save best model
        
        # Load best weights and evaluate on TEST set
        if best_state_dict is not None:
            model.load_state_dict(best_state_dict)
        
        # Evaluate on TEST set at early-stop iteration
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        # Evaluate training accuracy at early-stop
        _, train_acc = evaluate(model, train_loader, criterion, device)
        
        # Use the weights at early-stop for pruning (not final epoch weights)
        trained_state_dict = best_state_dict if best_state_dict is not None else copy.deepcopy(model.state_dict())
            
        # Calculate current sparsity stats from the actual mask
        total_params = 0
        remaining_params = 0
        
        for name, param in model.named_parameters():
            if 'weight' in name:
                total_params += param.numel()
                if name in mask:
                    remaining_params += mask[name].sum().item()
        
        remaining_percent = 100.0 * remaining_params / total_params
        sparsity = 100.0 - remaining_percent
        
        print(f"-> Result: Sparsity {sparsity:.1f}% | Test Acc @ Early-Stop: {test_acc:.2f}% | Train Acc: {train_acc:.2f}% | Early-Stop Epoch: {early_stop_epoch}")
        
        results.append({
            "model": model_name,
            "dropout": use_dropout,
            "type": experiment_type,
            "round": round_idx,
            "sparsity": sparsity,
            "remaining_percent": remaining_percent,
            "test_accuracy": test_acc,
            "train_accuracy": train_acc,
            "early_stop_epoch": early_stop_epoch,
            "early_stop_iteration": early_stop_epoch * len(train_loader)
        })

        # --- TIME ESTIMATION ---
        round_duration = time.time() - round_start
        rounds_left = iterations - (round_idx + 1)
        eta_seconds = rounds_left * round_duration
        eta_time = datetime.now() + timedelta(seconds=eta_seconds)

        print(f"Round {round_idx} took {round_duration/60:.1f}m. "
              f"Estimated model completion at: {eta_time.strftime('%H:%M:%S')}")

        # 4. Pruning Step: Create mask for NEXT round (iterative pruning)
        if round_idx < iterations - 1:
            # Calculate cumulative pruning percentages for NEXT round
            conv_cumulative = 1 - (1 - conv_prune_rate)**(round_idx + 1)
            fc_cumulative = 1 - (1 - fc_prune_rate)**(round_idx + 1)
            
            # Create temp model with trained weights to calculate new mask
            temp_model = model_class(use_dropout=use_dropout).to(device)
            temp_model.load_state_dict(trained_state_dict)
            
            # Calculate mask based on trained weight magnitudes
            mask = get_mask_layerwise(temp_model, conv_cumulative, fc_cumulative)

    return results

if __name__ == "__main__":
    # Create results directory if it doesn't exist
    os.makedirs("results", exist_ok=True)
    
    all_data = []
    
    configs = [
        # Without dropout
        ("Conv2", Conv2, False), 
        ("Conv4", Conv4, False),
        ("Conv6", Conv6, False),
        
        # With dropout (for Figure 6 comparison)
        ("Conv2", Conv2, True),
        ("Conv4", Conv4, True),
        ("Conv6", Conv6, True)
    ]
    
    types = ["winning_ticket", "random_reinit"]

    for m_name, m_class, d_out in configs:
        for t in types:
            print(f"\n{'='*80}")
            print(f"Starting: {m_name} | Dropout={d_out} | Type={t}")
            print(f"{'='*80}")
            
            # Run the experiment
            data = run_experiment(m_name, m_class, d_out, t, iterations=15, epochs=25)
            all_data.extend(data)
            
            # Save THIS config's results separately
            config_name = f"{m_name}_dropout{d_out}_{t}"
            config_df = pd.DataFrame(data)
            config_df.to_csv(f"results/{config_name}.csv", index=False)
            print(f"\nSaved results for {config_name} to results/{config_name}.csv")
            
            # Also update the cumulative backup
            pd.DataFrame(all_data).to_csv("results/all_results_cumulative.csv", index=False)

    # Final Save
    final_df = pd.DataFrame(all_data)
    final_df.to_csv("results/final_lottery_ticket_results.csv", index=False)
    print(f"\n{'='*80}")
    print("All experiments finished!")
    print(f"{'='*80}")
    print(f"Individual config results: results/<config_name>.csv")
    print(f"Cumulative results: results/all_results_cumulative.csv")
    print(f"Final results: results/final_lottery_ticket_results.csv")