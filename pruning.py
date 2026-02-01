import torch

def get_mask_layerwise(model, prune_percent_conv, prune_percent_fc):
    """
    Gives a mask for pruning the weights.
    """
    mask = {}
    
    for name, param in model.named_parameters():
        if 'weight' not in name:
            continue
            
        # Determines if this is a conv or fc layer based on name
        if 'features' in name:  # Conv layers have 'feature' in the name
            prune_percent = prune_percent_conv
        elif 'classifier' in name:  # FC layers have 'classifier' in the name
            prune_percent = prune_percent_fc
        else:
            prune_percent = prune_percent_fc  # Default to FC rate
        
        # Get threshold weight magnitude for this layer
        weights = param.data.abs().view(-1)
        threshold = torch.quantile(weights, prune_percent)
        mask[name] = (param.data.abs() > threshold).float()
    
    return mask


def apply_mask(model, mask):
    """Applies the mask to prune unwanted weights."""
    for name, param in model.named_parameters():
        if name in mask:
            param.data.mul_(mask[name])