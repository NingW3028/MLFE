import torch
import numpy as np
import time

def compute_dextr(net, inputs, targets):
    """
    Dextr: Zero-cost proxy combining SVD and curvature.
    
    Simplified version: uses SVD part only (curvature computation is too complex).
    
    Args:
        net: Neural network
        inputs: Input data
        targets: Target labels (unused)
    
    Returns:
        (score, time): Dextr score and computation time
    """
    start_time = time.time()
    
    device = next(net.parameters()).device
    inputs = inputs.to(device)
    
    svd_list = []
    
    def forward_hook(module, data_input, data_output):
        if isinstance(data_output, tuple):
            fea = data_output[0]
        else:
            fea = data_output
        
        fea = fea.clone().detach()
        n = fea.shape[0]
        fea = fea.reshape(n, -1)
        
        # Compute singular value ratio
        try:
            s = torch.linalg.svdvals(fea)
            svd_ratio = torch.min(s) / (torch.max(s) + 1e-10)
            if not torch.isnan(svd_ratio) and not torch.isinf(svd_ratio):
                svd_list.append(svd_ratio.item())
        except:
            pass
    
    # Register hooks
    hooks = []
    for module in net.modules():
        if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear, torch.nn.ReLU)):
            hooks.append(module.register_forward_hook(forward_hook))
    
    # Forward pass
    net.eval()
    with torch.no_grad():
        _ = net(inputs)
    
    # Remove hooks
    for hook in hooks:
        hook.remove()
    
    # Compute Dextr score
    if len(svd_list) > 0:
        svd_sum = sum(svd_list)
        score = np.log(1 + svd_sum)
    else:
        score = 0.0
    
    elapsed_time = time.time() - start_time
    return score, elapsed_time
