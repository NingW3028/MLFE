import torch
import numpy as np
import time
from scipy import stats as scipy_stats

def get_effective_rank(matrix):
    """Compute effective rank of a matrix - optimized version."""
    # Use torch SVD to avoid CPU/GPU transfers
    s = torch.linalg.svdvals(matrix)
    s_sum = torch.sum(s)
    if s_sum == 0:
        return 0.0
    s = s / s_sum
    # Only transfer to CPU when necessary
    s_np = s.detach().cpu().numpy()
    erank = np.e ** scipy_stats.entropy(s_np)
    return np.nan_to_num(erank)

def compute_near(net, inputs, targets):
    """
    NEAR: Network Expressivity by Activation Rank
    
    Evaluates network expressivity by computing effective rank of activation matrices.
    Optimized: reduced CPU/GPU data transfers, batch processing.
    
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
    
    def get_activation_hook(module, input, output):
        if isinstance(output, tuple):
            output = output[0]
        # Convert to 2D matrix
        if output.dim() > 2:
            output = torch.transpose(output, 1, 3).flatten(0, 2)
        elif output.dim() == 1:
            output = output.unsqueeze(0)
        activations.append(output.detach())
    
    # Register hooks
    hooks = []
    for module in net.modules():
        if hasattr(module, 'weight') or isinstance(module, (torch.nn.ReLU, torch.nn.Sigmoid, torch.nn.Tanh)):
            hooks.append(module.register_forward_hook(get_activation_hook))
    
    # Forward pass
    net.eval()
    with torch.no_grad():
        _ = net(inputs)
    
    # Remove hooks
    for hook in hooks:
        hook.remove()
    
    # Batch compute NEAR score
    score = 0.0
    for activation in activations:
        if activation.numel() == 0 or activation.shape[0] == 0 or activation.shape[1] == 0:
            continue
        try:
            erank = get_effective_rank(activation)
            score += erank
        except:
            pass
    
    elapsed_time = time.time() - start_time
    return score, elapsed_time
