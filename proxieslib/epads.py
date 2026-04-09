import torch
import numpy as np
import time

def compute_epads(net, inputs, targets, sigma=0.01):
    """
    ePADS: Efficient Perturbation-Aware Distinguishing Score
    
    Based on original implementation: ePADS-main/score_function/Product.py
    
    Args:
        net: Neural network
        inputs: Input data [N, C, H, W]
        targets: Target labels (unused)
        sigma: Gaussian noise standard deviation
    
    Returns:
        (score, time): ePADS score and computation time
    """
    start_time = time.time()
    
    device = next(net.parameters()).device
    inputs = inputs.to(device)
    batch_size = inputs.size(0)
    
    # Generate noisy input
    noise = torch.randn_like(inputs) * sigma
    inputs_noise = inputs + noise
    
    # Concatenate original and noisy inputs
    x2 = torch.cat([inputs, inputs_noise], 0)
    
    # Initialize accumulator
    net.diffs = 0
    
    def counting_forward_hook_act(module, inp, out):
        try:
            if isinstance(inp, tuple):
                inp = inp[0]
            
            # Split original and noisy features
            feature, feature_noise = torch.split(inp, [batch_size, batch_size], 0)
            
            # Compute ReLU activations
            xx = torch.gt(feature, 0).float()
            xx_shuffle = xx[torch.randperm(xx.size(0))]
            xx_noise = torch.gt(feature_noise, 0).float()
            
            # Compute Hamming distance
            diff1 = torch.sum(torch.abs(xx_shuffle - xx))
            diff2 = torch.sum(torch.abs(xx_noise - xx))
            
            # Accumulate product
            net.diffs += diff1 * diff2
            
        except Exception as e:
            pass
    
    # Register hooks on all ReLU layers
    hooks = []
    for name, module in net.named_modules():
        if 'ReLU' in str(type(module)):
            hooks.append(module.register_forward_hook(counting_forward_hook_act))
    
    # Forward pass
    net.eval()
    try:
        with torch.no_grad():
            _ = net(x2)
        
        score = float(net.diffs.cpu().item() if torch.is_tensor(net.diffs) else net.diffs)
    except Exception as e:
        score = 0.0
    finally:
        # Remove hooks
        for hook in hooks:
            hook.remove()
        # Cleanup
        if hasattr(net, 'diffs'):
            delattr(net, 'diffs')
    
    elapsed_time = time.time() - start_time
    return score, elapsed_time
