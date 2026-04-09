import torch
import numpy as np
import time

def compute_wrcor(net, inputs, targets, split_data=1):
    """
    WRCor (Weighted Correlation) proxy
    Returns: (score, time)
    """
    start_time = time.time()
    N = inputs.size(0)
    
    def get_counting_forward_hook(weight):
        def counting_forward_hook(module, inp, out):
            try:
                if isinstance(inp, tuple):
                    inp = inp[0]
                inp = inp.view(inp.size(0), -1)
                act_corrs = np.corrcoef(inp.detach().cpu().numpy())
                if np.sum(np.isnan(act_corrs)) == 0:
                    net.K = net.K + weight * (np.abs(act_corrs))
            except:
                pass
        return counting_forward_hook
    
    def get_counting_backward_hook(weight):
        def counting_backward_hook(module, grad_input, grad_output):
            try:
                if isinstance(grad_input, tuple):
                    grad_input = grad_input[0]
                grad_input = grad_input.view(grad_input.size(0), -1)
                grad_corrs = np.corrcoef(grad_input.detach().cpu().numpy())
                if np.sum(np.isnan(grad_corrs)) == 0:
                    net.K = net.K + weight * (np.abs(grad_corrs))
            except:
                pass
        return counting_backward_hook
    
    def hooklogdet(K):
        s, ld = np.linalg.slogdet(K)
        if s <= 0 or np.isinf(ld) or np.isnan(ld):
            return 0.0
        return ld
    
    # Find modules
    modules = None
    if hasattr(net, 'cells'):
        modules = net.cells
    elif hasattr(net, 'layers'):
        modules = net.layers
    elif hasattr(net, 'block_list'):
        modules = net.block_list
    
    # Register hooks and store handles
    hook_handles = []
    if modules:
        for i, module in enumerate(modules):
            for name, m in module.named_modules():
                if 'ReLU' in str(type(m)):
                    hook_handles.append(m.register_forward_hook(get_counting_forward_hook(2**i)))
                    hook_handles.append(m.register_backward_hook(get_counting_backward_hook(2**i)))
        
        if hasattr(net, 'lastact'):
            for name, m in net.lastact.named_modules():
                if 'ReLU' in str(type(m)):
                    hook_handles.append(m.register_forward_hook(get_counting_forward_hook(2**len(modules))))
                    hook_handles.append(m.register_backward_hook(get_counting_backward_hook(2**len(modules))))
    
    # Compute score
    s = []
    loss_fn = torch.nn.CrossEntropyLoss()
    
    for sp in range(split_data):
        net.zero_grad()
        net.K = np.zeros((N//split_data, N//split_data))
        
        st = sp * N // split_data
        en = (sp + 1) * N // split_data
        
        outputs = net(inputs[st:en])
        if isinstance(outputs, tuple):
            outputs = outputs[0]
        
        loss = loss_fn(outputs, targets[st:en])
        loss.backward()
        
        s.append(hooklogdet(net.K))
    
    if len(s) == 0 or all(x == 0 for x in s):
        wrcor_score = 0.0
    else:
        wrcor_score = np.sum(s)
    elapsed_time = time.time() - start_time
    
    # Remove hooks
    for handle in hook_handles:
        handle.remove()
    
    return wrcor_score, elapsed_time
