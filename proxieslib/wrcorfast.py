import torch
import numpy as np
import time

def compute_wrcorfast(net, inputs, targets, split_data=1):
    """
    WRCorFast - Optimized version of WRCor
    Returns: (score, time)
    """
    start_time = time.time()
    N = inputs.size(0)
    
    def get_counting_forward_hook(weight):
        def counting_forward_hook(module, inp, out):
            try:
                if isinstance(inp, tuple):
                    inp = inp[0]
                inp = inp.view(inp.size(0), -1).detach().cpu().numpy()
                # Use only diagonal and upper triangle to save computation
                act_corrs = np.abs(np.corrcoef(inp))
                if not np.any(np.isnan(act_corrs)):
                    net.K += weight * act_corrs
            except:
                pass
        return counting_forward_hook
    
    def get_counting_backward_hook(weight):
        def counting_backward_hook(module, grad_input, grad_output):
            try:
                if isinstance(grad_input, tuple):
                    grad_input = grad_input[0]
                grad_input = grad_input.view(grad_input.size(0), -1).detach().cpu().numpy()
                grad_corrs = np.abs(np.corrcoef(grad_input))
                if not np.any(np.isnan(grad_corrs)):
                    net.K += weight * grad_corrs
            except:
                pass
        return counting_backward_hook
    
    # Find modules
    modules = None
    if hasattr(net, 'cells'):
        modules = net.cells
    elif hasattr(net, 'layers'):
        modules = net.layers
    elif hasattr(net, 'block_list'):
        modules = net.block_list
    
    # Register hooks only on key layers
    if modules:
        for i, module in enumerate(modules):
            for name, m in module.named_modules():
                if 'ReLU' in str(type(m)):
                    m.register_forward_hook(get_counting_forward_hook(2**i))
                    m.register_backward_hook(get_counting_backward_hook(2**i))
        
        if hasattr(net, 'lastact'):
            for name, m in net.lastact.named_modules():
                if 'ReLU' in str(type(m)):
                    m.register_forward_hook(get_counting_forward_hook(2**len(modules)))
                    m.register_backward_hook(get_counting_backward_hook(2**len(modules)))
    
    # Single pass computation
    net.zero_grad()
    net.K = np.zeros((N, N))
    
    outputs = net(inputs)
    if isinstance(outputs, tuple):
        outputs = outputs[0]
    
    loss = torch.nn.functional.cross_entropy(outputs, targets)
    loss.backward()
    
    # Fast log determinant using sign
    s, ld = np.linalg.slogdet(net.K)
    wrcor_score = ld if s > 0 else 0.0
    
    elapsed_time = time.time() - start_time
    
    return wrcor_score, elapsed_time
