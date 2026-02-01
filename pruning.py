import torch

def get_mask_layerwise(model, prune_percent_conv, prune_percent_fc):
    """
    LAYER-WISE pruning: Prunes each layer independently at specified rates.
    Used for Conv-2, Conv-4, Conv-6 (Section 3 of paper).
    
    Args:
        model: The neural network
        prune_percent_conv: Cumulative percentage to prune from convolutional layers
        prune_percent_fc: Cumulative percentage to prune from fully-connected layers
    
    Returns:
        Dictionary of masks for each weight tensor
    """
    mask = {}
    
    for name, param in model.named_parameters():
        if 'weight' not in name:
            continue
            
        # Determine if this is a conv or fc layer based on module name
        if 'features' in name:  # Conv layers are in 'features' module
            prune_percent = prune_percent_conv
        elif 'classifier' in name:  # FC layers are in 'classifier' module
            prune_percent = prune_percent_fc
        else:
            prune_percent = prune_percent_fc  # Default to FC rate
        
        # Get threshold for this specific layer
        weights = param.data.abs().view(-1)
        threshold = torch.quantile(weights, prune_percent)
        mask[name] = (param.data.abs() > threshold).float()
    
    return mask


def apply_mask(model, mask):
    """Applies the mask to model parameters (sets pruned weights to zero)."""
    for name, param in model.named_parameters():
        if name in mask:
            param.data.mul_(mask[name])