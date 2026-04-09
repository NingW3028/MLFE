# Copyright 2021 Samsung Electronics Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================

import torch
from torch import nn
import numpy as np
import time

from . import measure


def network_weight_gaussian_init(net: nn.Module):
    with torch.no_grad():
        for n, m in net.named_modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                try:
                    nn.init.ones_(m.weight)
                    nn.init.zeros_(m.bias)
                except:
                    pass
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.zeros_(m.bias)
            else:
                continue

    return net


def get_zen(gpu, model, mixup_gamma=1e-2, resolution=32, batch_size=64, repeat=32,
                      fp16=False):
    info = {}
    nas_score_list = []
    if gpu is not None:
        device = torch.device(gpu)
    else:
        device = torch.device('cpu')

    if fp16:
        dtype = torch.half
    else:
        dtype = torch.float32

    # Check if forward_pre_GAP method exists, otherwise use regular forward
    if hasattr(model, 'forward_pre_GAP'):
        forward_fn = model.forward_pre_GAP
    else:
        forward_fn = model.forward

    with torch.no_grad():
        for repeat_count in range(repeat):
            network_weight_gaussian_init(model)
            input = torch.randn(size=[batch_size, 3, resolution, resolution], device=device, dtype=dtype)
            input2 = torch.randn(size=[batch_size, 3, resolution, resolution], device=device, dtype=dtype)
            mixup_input = input + mixup_gamma * input2
            output = forward_fn(input)
            mixup_output = forward_fn(mixup_input)

            # Dynamically choose sum dimensions based on output shape
            diff = torch.abs(output - mixup_output)
            if diff.dim() == 4:  # (B, C, H, W)
                nas_score = torch.sum(diff, dim=[1, 2, 3])
            elif diff.dim() == 3:  # (B, C, L) 
                nas_score = torch.sum(diff, dim=[1, 2])
            elif diff.dim() == 2:  # (B, C) - classification output
                nas_score = torch.sum(diff, dim=1)
            else:
                nas_score = torch.sum(diff)
            nas_score = torch.mean(nas_score)

            # compute BN scaling
            log_bn_scaling_factor = 0.0
            for m in model.modules():
                if isinstance(m, nn.BatchNorm2d):
                    try:
                        bn_scaling_factor = torch.sqrt(torch.mean(m.running_var))
                        log_bn_scaling_factor += torch.log(bn_scaling_factor)
                    except:
                        pass
                pass
            pass
            nas_score = torch.log(nas_score) + log_bn_scaling_factor
            nas_score_list.append(float(nas_score))

    std_nas_score = np.std(nas_score_list)
    avg_precision = 1.96 * std_nas_score / np.sqrt(len(nas_score_list))
    avg_nas_score = np.mean(nas_score_list)

    info = float(avg_nas_score)
    return info





@measure('zen', bn=True)
def compute_zen(net, inputs, targets, split_data=1, loss_fn=None):
    device = inputs.device
    # Compute gradients (but don't apply them)
    net.zero_grad()

    time_beg = time.time()
    try:
        zen = get_zen(device, net)
    except Exception as e:
        zen = np.nan

    return zen
