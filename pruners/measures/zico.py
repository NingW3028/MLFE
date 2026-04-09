# Copyright 2021 Samsung Electronics Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
import time

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================

import numpy as np
import torch

from . import measure
from torch import nn


def getgrad(model: torch.nn.Module, grad_dict: dict, step_iter=0):
    """Collect gradients from Conv2d and Linear layers in the model."""
    for name, mod in model.named_modules():
        if isinstance(mod, (nn.Conv2d, nn.Linear)):
            if mod.weight.grad is None:
                continue
            grad_data = mod.weight.grad.data.clone().reshape(-1)
            if step_iter == 0:
                grad_dict[name] = [grad_data]
            else:
                if name in grad_dict:
                    grad_dict[name].append(grad_data)
    return grad_dict


def caculate_zico(grad_dict):
    """Compute ZICO score."""
    nsr_mean_sum_abs = 0
    
    for modname, grads in grad_dict.items():
        # Compute on GPU to avoid frequent CPU transfers
        grad_tensor = torch.stack(grads)  # (split_data, num_params)
        
        # Compute std and mean
        nsr_std = torch.std(grad_tensor, dim=0)
        nsr_mean_abs = torch.mean(torch.abs(grad_tensor), dim=0)
        
        # Find indices with non-zero std
        nonzero_mask = nsr_std > 1e-10
        if nonzero_mask.sum() == 0:
            continue
            
        # Compute ratio
        ratio = nsr_mean_abs[nonzero_mask] / nsr_std[nonzero_mask]
        tmpsum = torch.sum(ratio)
        
        if tmpsum > 0:
            nsr_mean_sum_abs += torch.log(tmpsum).item()
    
    return nsr_mean_sum_abs


def getzico(network, inputs, targets, loss_fn, split_data=2):
    """Compute ZICO metric."""
    grad_dict = {}
    network.train()
    device = inputs.device
    N = inputs.shape[0]

    for sp in range(split_data):
        network.zero_grad()
        st = sp * N // split_data
        en = (sp + 1) * N // split_data
        outputs = network.forward(inputs[st:en])
        loss = loss_fn(outputs, targets[st:en])
        loss.backward()
        grad_dict = getgrad(network, grad_dict, sp)
    
    res = caculate_zico(grad_dict)
    return res





@measure('zico', bn=True)
def compute_zico(net, inputs, targets, split_data=2, loss_fn=None):

    # Compute gradients (but don't apply them)
    net.zero_grad()

    try:
        time_beg = time.time()
        zico = getzico(net, inputs, targets, loss_fn, split_data=split_data)
        time_end = time.time()
    except Exception as e:
        zico= np.nan
    return zico