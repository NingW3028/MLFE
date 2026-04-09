import torch
import numpy as np
import time
from scipy import stats as scipy_stats

def get_effective_rank_fast(matrix):
    """Fast effective rank computation using approximation."""
    # Limit matrix size to speed up SVD
    max_size = 512
    if matrix.shape[0] > max_size:
        indices = torch.randperm(matrix.shape[0])[:max_size]
        matrix = matrix[indices]
    if matrix.shape[1] > max_size:
        indices = torch.randperm(matrix.shape[1])[:max_size]
        matrix = matrix[:, indices]
    
    s = torch.linalg.svdvals(matrix)
    s = s / torch.sum(s)
    erank = torch.e ** scipy_stats.entropy(s.detach().cpu().numpy())
    return np.nan_to_num(erank)

def compute_near_fast(net, inputs, targets):
    """
    NEAR Fast: Accelerated Network Expressivity by Activation Rank.
    
    Optimization strategies:
    1. Sample key layers only (every 2nd layer)
    2. Limit activation matrix size
    3. Use faster SVD computation
    
    Args:
        net: Neural network
        inputs: Input data
        targets: Target labels (unused)
    
    Returns:
        (score, time): NEAR score and computation time
    """
    start_time = time.time()
    
    device = next(net.parameters()).device
    inputs = inputs.to(device)
    
    activations = []
    hooks = []
    layer_count = 0
    
    def get_activation_hook(storage):
        def hook(module, input, output):
            if isinstance(output, tuple):
                output = output[0]
            # Convert to 2D matrix
            if output.dim() > 2:
                output = torch.transpose(output, 1, 3).flatten(0, 2)
            storage.append(output.detach())
        return hook
    
    # Register hooks on key layers only (every 2nd layer)
    for module in net.modules():
        if hasattr(module, 'weight') or isinstance(module, (torch.nn.ReLU, torch.nn.Sigmoid, torch.nn.Tanh)):
            if layer_count % 2 == 0:  # Only sample even layers
                hooks.append(module.register_forward_hook(get_activation_hook(activations)))
            layer_count += 1
    
    # Forward pass
    net.eval()
    with torch.no_grad():
        _ = net(inputs)
    
    # Remove hooks
    for hook in hooks:
        hook.remove()
    
    # Compute NEAR score
    score = 0.0
    for activation in activations:
        if len(activation) == 0 or activation.numel() == 0:
            continue
        
        # Ensure 2D matrix
        if activation.dim() == 1:
            activation = activation.unsqueeze(0)
        
        # Only compute effective rank when matrix is large enough
        if activation.shape[0] > 0 and activation.shape[1] > 0:
            try:
                erank = get_effective_rank_fast(activation)
                score += erank
            except:
                pass
    
    elapsed_time = time.time() - start_time
    return score, elapsed_time
