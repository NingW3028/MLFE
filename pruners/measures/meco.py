import copy
import time
import torch.nn.functional as F
import numpy as np
import torch
from torch import nn

from . import measure


def get_score(net, x, target, device, split_data):
    result_list = []
    hooks = []  # Store hook handles for later removal

    def forward_hook(module, data_input, data_output):
        # Only process meaningful output
        if isinstance(data_output, tuple):
            fea = data_output[0]
        else:
            fea = data_output
        
        if fea is None or fea.dim() < 2:
            return
            
        fea = fea.detach()
        fea = fea.reshape(fea.shape[0], -1)
        
        # Skip too small features (speed up)
        if fea.shape[1] < 2:
            return
            
        # deal with nan and inf
        corr = torch.corrcoef(fea)
        corr[torch.isnan(corr)] = 0
        corr[torch.isinf(corr)] = 0
        
        # Use eigvalsh for faster computation (symmetric matrix)
        try:
            values = torch.linalg.eigvalsh(corr)
            result = torch.min(values)
            result_list.append(result)
        except:
            pass

    # Register hooks on key layers only (speed up)
    for name, module in net.named_modules():
        if isinstance(module, (nn.Linear, nn.Conv2d, nn.LayerNorm)):
            h = module.register_forward_hook(forward_hook)
            hooks.append(h)

    N = x.shape[0]
    with torch.no_grad():  # No gradient needed, speed up
        y = net(x)
    
    # Remove all hooks (important!)
    for h in hooks:
        h.remove()
    
    if len(result_list) == 0:
        return 0.0
        
    results = torch.stack(result_list)
    results = results[torch.logical_not(torch.isnan(results))]
    v = torch.sum(results)
    return v.item()


@measure('meco', bn=True)
def compute_meco(net, inputs, targets, split_data=1, loss_fn=F.cross_entropy):
    device = inputs.device
    # Compute gradients (but don't apply them)
    net.zero_grad()

    time_beg = time.time()
    try:
        meco = get_score(net, inputs, targets, device, split_data=split_data)
    except Exception as e:
        meco = np.nan
    return meco
