import torch
import numpy as np
import time

def compute_wrcorultrafast(net, inputs, targets, split_data=1):
    """
    WRCorUltraFast - Ultra optimized version
    - Sample only key layers
    - Use smaller correlation matrix
    - Skip gradient correlation
    Returns: (score, time)
    """
    start_time = time.time()
    N = min(inputs.size(0), 32)  # Limit batch size for speed
    inputs = inputs[:N]
    targets = targets[:N]
    
    net.K = np.zeros((N, N))
    
    def get_counting_forward_hook(weight):
        def counting_forward_hook(module, inp, out):
            try:
                if isinstance(inp, tuple):
                    inp = inp[0]
                inp = inp.view(inp.size(0), -1)[:N].detach().cpu().numpy()
                # Fast correlation using only mean centering
                inp_centered = inp - inp.mean(axis=1, keepdims=True)
                corr = np.abs(np.dot(inp_centered, inp_centered.T))
                net.K += weight * corr
            except:
                pass
        return counting_forward_hook
    
    # Find modules
    modules = None
    if hasattr(net, 'cells'):
        modules = net.cells
    elif hasattr(net, 'layers'):
        modules = net.layers
    elif hasattr(net, 'block_list'):
        modules = net.block_list
    
    # Register hooks only on every 2nd layer to save time
    if modules:
        for i in range(0, len(modules), 2):  # Skip every other layer
            module = modules[i]
            for name, m in module.named_modules():
                if 'ReLU' in str(type(m)):
                    m.register_forward_hook(get_counting_forward_hook(2**i))
                    break  # Only first ReLU per module
    
    # Single forward pass only (no backward)
    with torch.no_grad():
        outputs = net(inputs)
        if isinstance(outputs, tuple):
            outputs = outputs[0]
    
    # Fast determinant approximation
    try:
        # Add small regularization for stability
        net.K += np.eye(N) * 1e-6
        sign, logdet = np.linalg.slogdet(net.K)
        score = logdet if sign > 0 else 0.0
    except:
        score = 0.0
    
    elapsed_time = time.time() - start_time
    return score, elapsed_time
