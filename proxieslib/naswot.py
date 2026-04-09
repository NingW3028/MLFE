import numpy as np
import torch

def compute_naswot(net, inputs, dtype, split_data=1, loss_fn=None, benchtype='tss', dataset='cifar10'):
    """
    NASWOT (Neural Architecture Search Without Training) proxy
    Computes log determinant of kernel matrix from ReLU activation patterns
    """
    device = inputs.device
    batch_size = inputs.size(0)
    
    net.K = np.zeros((batch_size, batch_size))
    
    def counting_forward_hook(module, inp, out):
        try:
            if not hasattr(module, 'visited_backwards') or not module.visited_backwards:
                return
            if isinstance(inp, tuple):
                inp = inp[0]
            inp = inp.view(inp.size(0), -1)
            x_act = (inp > 0).float()
            K = x_act @ x_act.t()
            K2 = (1. - x_act) @ (1. - x_act.t())
            net.K = net.K + K.cpu().numpy() + K2.cpu().numpy()
        except Exception as e:
            pass
    
    def counting_backward_hook(module, inp, out):
        module.visited_backwards = True
    
    hooks = []
    for name, module in net.named_modules():
        if 'ReLU' in str(type(module)):
            module.visited_backwards = False
            hooks.append(module.register_forward_hook(counting_forward_hook))
            hooks.append(module.register_backward_hook(counting_backward_hook))
    
    net.zero_grad()
    inputs.requires_grad_(True)
    
    try:
        if benchtype == 'tss':
            output = net(inputs)
            if isinstance(output, tuple):
                logits = output[0]
            else:
                logits = output
        else:
            logits = net(inputs)
        
        logits.backward(torch.ones_like(logits))
        
        net.zero_grad()
        inputs.grad = None
        
        _ = net(inputs)
        
        s, ld = np.linalg.slogdet(net.K)
        
        if s <= 0 or np.isinf(ld) or np.isnan(ld):
            score = 0.0
        else:
            score = ld
        
    except Exception as e:
        score = 0.0
    finally:
        for hook in hooks:
            hook.remove()
        for name, module in net.named_modules():
            if hasattr(module, 'visited_backwards'):
                delattr(module, 'visited_backwards')
        if hasattr(net, 'K'):
            delattr(net, 'K')
        inputs.requires_grad_(False)
    
    return score
