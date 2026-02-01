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
    """
    for layer in model.modules():
        if isinstance(layer, (nn.Conv2d, nn.Linear)):
            nn.init.xavier_normal_(layer.weight)
            if layer.bias is not None:
                nn.init.constant_(layer.bias, 0)

def run_quick_test(model_name, model_class, use_dropout, experiment_type, iterations=3, epochs=2):
    """Quick test with minimal iterations and epochs"""
    # Determine per-iteration pruning rates
    if model_name == "Conv2" or model_name == "Conv4":
        conv_prune_rate = 0.10
        fc_prune_rate = 0.20
    elif model_name == "Conv6":
        conv_prune_rate = 0.15
        fc_prune_rate = 0.20
    else:
        conv_prune_rate = 0.20
        fc_prune_rate = 0.20
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    train_loader, val_loader, _ = get_loaders(batch_size=60) 
    
    # Setup Model and capture Initial State
    model = model_class(use_dropout=use_dropout).to(device)
    initial_state_dict = copy.deepcopy(model.state_dict())
    
    # Initialize mask as all ones
    mask = {name: torch.ones_like(param) for name, param in model.named_parameters() if 'weight' in name}
    
    results = []

    for round_idx in range(iterations):
        round_start = time.time()
        
        # Reset to theta_0 or reinitialize
        if experiment_type == "winning_ticket":
            model.load_state_dict(copy.deepcopy(initial_state_dict))
        else:
            reinitialize_model(model)
            
        apply_mask(model, mask)
        
        lr = 0.0003 if model_name == "Conv2" else 0.0002
        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        
        best_acc = 0.0
        
        print(f"\n>>> Round {round_idx}/{iterations-1} | {experiment_type}")
        
        for epoch in range(1, epochs + 1):
            train_loss = train(model, train_loader, optimizer, criterion, device, mask)
            _, val_acc = evaluate(model, val_loader, criterion, device)
            
            if val_acc > best_acc:
                best_acc = val_acc
            
            print(f"  Epoch {epoch}/{epochs}: Loss={train_loss:.4f}, Val Acc={val_acc:.2f}%")
            
        # Calculate sparsity
        total_params = 0
        remaining_params = 0
        
        for name, param in model.named_parameters():
            if 'weight' in name:
                total_params += param.numel()
                if name in mask:
                    remaining_params += mask[name].sum().item()
        
        remaining_percent = 100.0 * remaining_params / total_params
        sparsity = 100.0 - remaining_percent
        
        print(f"  -> Sparsity: {sparsity:.1f}% | Remaining: {remaining_percent:.1f}% | Best Acc: {best_acc:.2f}%")
        
        results.append({
            "model": model_name,
            "dropout": use_dropout,
            "type": experiment_type,
            "round": round_idx,
            "sparsity": sparsity,
            "remaining_percent": remaining_percent,
            "test_accuracy": best_acc
        })

        # Pruning for next round: Calculate cumulative pruning (iterative)
        if round_idx < iterations - 1:
            # Calculate cumulative pruning for NEXT round
            conv_cumulative = 1 - (1 - conv_prune_rate)**(round_idx + 1)
            fc_cumulative = 1 - (1 - fc_prune_rate)**(round_idx + 1)
            
            print(f"  -> Next round cumulative pruning: Conv={conv_cumulative*100:.1f}%, FC={fc_cumulative*100:.1f}%")
            
            # CRITICAL: Calculate mask from the TRAINED weights (before reset)
            # We prune based on current trained magnitudes, not the masked values
            mask = get_mask_layerwise(model, conv_cumulative, fc_cumulative)

    return results

if __name__ == "__main__":
    print("🛠️ STARTING SANITY CHECK...")
    all_data = []
    
    # Test with just one configuration
    # Use 3 iterations and 2 epochs for speed
    configs = [("Conv2", Conv2, False)]
    types = ["winning_ticket"]

    for m_name, m_class, d_out in configs:
        for t in types:
            data = run_quick_test(m_name, m_class, d_out, t, iterations=3, epochs=2)
            all_data.extend(data)

    # Check if the CSV saves correctly
    test_filename = "sanity_check_results.csv"
    final_df = pd.DataFrame(all_data)
    final_df.to_csv(test_filename, index=False)
    
    if os.path.exists(test_filename):
        print(f"\n✅ SUCCESS: CSV saved to {test_filename}")
        print("\nResults Preview:")
        print(final_df.to_string())
        
        # Verify sparsity is increasing
        sparsities = final_df['sparsity'].tolist()
        print(f"\nSparsity progression: {sparsities}")
        if len(sparsities) > 1 and all(sparsities[i] < sparsities[i+1] for i in range(len(sparsities)-1)):
            print("✅ Sparsity is increasing correctly!")
        else:
            print("⚠️ Warning: Sparsity progression looks unusual")
    else:
        print("\n❌ ERROR: CSV was not created.")
