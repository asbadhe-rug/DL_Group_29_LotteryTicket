import torch

def get_mask(model, prune_percent):
    """
    Finds the global threshold for the bottom p% of weights
    and returns a dictionary of masks.
    """
    all_weights = []
    for name, param in model.named_parameters():
        if 'weight' in name:
            all_weights.append(param.data.abs().view(-1))
    
    all_weights = torch.cat(all_weights)
    threshold = torch.quantile(all_weights, prune_percent)
    
    mask = {}
    for name, param in model.named_parameters():
        if 'weight' in name:
            mask[name] = (param.data.abs() > threshold).float()
    return mask

def apply_mask(model, mask):
    for name, param in model.named_parameters():
        if name in mask:
            param.data.mul_(mask[name])