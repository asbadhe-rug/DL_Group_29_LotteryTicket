import torch
import torch.nn as nn

def get_mask(model, conv_rate=0.1, fc_rate=0.2):
    mask = {}
    
    # Get all weight-bearing modules in a list to find the last one
    modules = [(n, m) for n, m in model.named_modules() if isinstance(m, (nn.Conv2d, nn.Linear))]
    last_module_name = modules[-1][0]

    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            weight_name = f"{name}.weight"
            param = getattr(module, 'weight')
            
            # 1. Output Layer: use conv_rate (10%) per paper
            if name == last_module_name:
                rate = conv_rate
            # 2. Conv Layers: use conv_rate (10%)
            elif isinstance(module, nn.Conv2d):
                rate = conv_rate
            # 3. Hidden FC Layers: use fc_rate (20%)
            else:
                rate = fc_rate

            # Thresholding logic...
            weights = param.data.abs().view(-1)
            active_weights = weights[weights > 0]
            
            if len(active_weights) > 0:
                threshold = torch.quantile(active_weights, rate)
                mask[weight_name] = (param.data.abs() > threshold).float()
            else:
                mask[weight_name] = torch.zeros_like(param.data)
                
    return mask

def apply_mask(model, mask):
    """
    Multiplies model weights by the mask to zero out pruned connections.
    """
    for name, param in model.named_parameters():
        if name in mask:
            with torch.no_grad():
                param.data.mul_(mask[name])