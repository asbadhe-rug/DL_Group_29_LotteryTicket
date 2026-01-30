import torch
import torch.nn as nn
import torch.optim as optim
import copy
import pandas as pd
import os
import time
from datetime import datetime, timedelta
from run_experiments import run_experiment

# Local file imports
from models import Conv2, Conv4, Conv6 
from pruning import get_mask, apply_mask
from train import train, evaluate
from utils import get_loaders

if __name__ == "__main__":
    print("🛠️ STARTING SANITY CHECK...")
    all_data = []
    
    # 1. Test with just one configuration
    # 2. Set iterations=2 (to test if pruning actually happens)
    # 3. Set epochs=1 (to make it fast)
    configs = [("Conv2", Conv2, False)]
    types = ["winning_ticket"]

    for m_name, m_class, d_out in configs:
        for t in types:
            # We run 2 iterations to ensure get_mask and apply_mask work in sequence
            data = run_experiment(m_name, m_class, d_out, t, iterations=2, epochs=1)
            all_data.extend(data)

    # 2. Check if the CSV actually saves
    test_filename = "sanity_check_results.csv"
    final_df = pd.DataFrame(all_data)
    final_df.to_csv(test_filename, index=False)
    
    if os.path.exists(test_filename):
        print(f"\n✅ SUCCESS: CSV saved to {test_filename}")
        print(final_df.head())
    else:
        print("\n❌ ERROR: CSV was not created.")