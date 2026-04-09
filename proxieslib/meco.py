# Copyright 2021 Samsung Electronics Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
import copy
import time
import torch.nn.functional as F

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================

import numpy as np
import torch
from torch import nn


def get_score(net, x, target, device, split_data):
    result_list = []
    hooks = []  # Store hook handles for later removal

    def forward_hook(module, data_input, data_output):
        fea = data_output[0].detach()
        fea = fea.reshape(fea.shape[0], -1)
        
        # Skip layers with too small feature dimension (avoid meaningless computation)
        if fea.shape[1] < 2:
            return
            
        # deal with nan and inf
        corr = torch.corrcoef(fea)
        corr[torch.isnan(corr)] = 0
        corr[torch.isinf(corr)] = 0
        
        # Use eigvalsh instead of eig (faster for symmetric matrices, correlation matrix is symmetric)
        # eigvalsh only computes eigenvalues, not eigenvectors, which is faster
        values = torch.linalg.eigvalsh(corr)
        
        # Use real values directly (eigvalsh returns real numbers)
        result = torch.min(values)
        result_list.append(result)

    # Register hooks and save handles
    for name, modules in net.named_modules():
        hook = modules.register_forward_hook(forward_hook)
        hooks.append(hook)

    # Use torch.no_grad() to speed up forward pass (no gradient computation needed)
    with torch.no_grad():
        N = x.shape[0]
        for sp in range(split_data):
            st = sp * N // split_data
            en = (sp + 1) * N // split_data
            y = net(x[st:en])
    
    # Remove all hooks (important! avoid affecting subsequent computation)
    for hook in hooks:
        hook.remove()
    
    results = torch.tensor(result_list, device=device)  # Create tensor directly on GPU
    results = results[torch.logical_not(torch.isnan(results))]
    v = torch.sum(results)
    result_list.clear()
    return v.item()



def compute_meco(net, inputs, targets, split_data=1, loss_fn=F.cross_entropy):
    device = inputs.device
    # Compute gradients (but don't apply them)
    net.zero_grad()

    #try:
    time_beg=time.time()
    meco = get_score(net, inputs, targets, device, split_data=split_data)
    time_end=time.time()
    #print('meco_time:',time_end-time_beg)
    t=time_end-time_beg
    #except Exception as e:
    #    print(e)
    #    meco = np.nan
    #    t=np.nan
    return meco,t
