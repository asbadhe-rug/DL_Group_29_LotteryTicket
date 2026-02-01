import torch
import torch.nn as nn

def get_mask(model, conv_rate=0.1, fc_rate=0.2):
    """
    Creates a layer-wise mask.
    - Convolutional layers: pruned at conv_rate (10% per iteration)
    - Fully Connected layers: pruned at fc_rate (20% per iteration)
    """
    mask = {}
    for name, module in model.named_modules():
        # Check if the module is Conv or Linear
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            # Get the weight parameter name (usually 'layer_name.weight')
            weight_name = f"{name}.weight"
            param = getattr(module, 'weight')
            
            # Determine the rate (Paper uses 10% for Conv/Output, 20% for FC)
            # Usually the last Linear layer is treated as 'Output'
            is_output = (name == list(model.named_modules())[-1][0])
            rate = conv_rate if (isinstance(module, nn.Conv2d) or is_output) else fc_rate
            
            # 1. Get current weights that are NOT already pruned (non-zero)
            # This is important for ITERATIVE pruning
            weights = param.data.abs().view(-1)
            active_weights = weights[weights > 0]
            
            if len(active_weights) > 0:
                # 2. Find threshold for the bottom X% of ACTIVE weights
                threshold = torch.quantile(active_weights, rate)
                
                # 3. Create mask: 1 if weight is above threshold AND was already 1
                new_mask = (param.data.abs() > threshold).float()
                mask[weight_name] = new_mask
            else:
                mask[weight_name] = torch.zeros_like(param.data)
                
    return mask