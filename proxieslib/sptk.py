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
import torch.nn as nn
import numpy as np
import copy

def get_norm(p):
    p2 = abs(p.detach().cpu().numpy())
    p1 = np.zeros((p2.shape[0],p.shape[1]))
    for i in range(p2.shape[0]):
        for j in range(p2.shape[1]):
            p1[i][j] = np.sum(np.absolute(p2[i][j]))
    return p1

def kernel_probability(filter):

    for i in range(filter.shape[0]):
        s = np.sum(np.absolute(filter[i]))
        filter[i] = filter[i]/float(s)
    return filter

def generate_probability(network, verbose = False):
    net = copy.deepcopy(network)
    prob = []
    reverse_prob = []
    kernel_prob = []
    layer_no = 0
    for p in net.parameters():
        if len(p.data.size()) != 1:
            prob.append([])
            reverse_prob.append([])
            kernel_prob.append([])
            p2 = abs(p.detach().cpu().numpy())
            p1 = get_norm(p)
            p1 = np.array(p1)
            for i in range(p1.shape[0]):
                par = p1[i]
                if verbose == True:
                    print(f'i value : {i},' \
                          + f'\tTarget Value: {p1.shape[0]}', end="\r", flush=True)
                if sum(par) != 0:
                    pvals = [float(float(o) / float(sum(par))) for o in par]
                else:
                    pvals = [float(float(1) / float(par.shape[0])) for o in par]
                reverse_prob[layer_no].append(pvals)
            p1 = np.transpose(np.array(p1))
            for i in range(p1.shape[0]):
                par = p1[i]
                if verbose == True:
                    print(f'i value : {i},' \
                          + f'\tTarget Value: {p1.shape[0]}', end="\r", flush=True)
                if sum(par) != 0:
                    pvals = [float(float(o) / float(sum(par))) for o in par]
                else:
                    pvals = [float(float(1)/float(par.shape[0])) for o in par]
                prob[layer_no].append(pvals)
            for i in range(p2.shape[0]):
                kernel_prob[layer_no].append(kernel_probability(p2[i]))
            layer_no = layer_no + 1
    return prob, reverse_prob, kernel_prob


def recal_bn(network, inputs, targets, recalbn, device):
    for m in network.modules():
        if isinstance(m, torch.nn.BatchNorm2d):
            m.running_mean.data.fill_(0)
            m.running_var.data.fill_(0)
            m.num_batches_tracked.data.zero_()
            m.momentum = None
    network.train()
    with torch.no_grad():
        for i, (inputs, targets) in enumerate(zip(inputs, targets)):
            if i >= recalbn: break
            inputs = inputs.cuda(device=device, non_blocking=True)
            _, _ = network(inputs)
    return network


def get_sptk_n(inputs, targets, network, device, recalbn=0, train_mode=False, num_batch=1):
    prob, reverse_prob, kernel_prob =generate_probability(network, verbose = False)
    #print('prob', len(prob))
    #print('reverse_prob', len(reverse_prob))
    #print('kernel_prob', len(kernel_prob))
    #print('prob', prob[0])
    #print('reverse_prob', (reverse_prob[0]))
    #print('kernel_prob', (kernel_prob[0][0]))
    return conds






def compute_sptk(net, inputs, targets, split_data=1, loss_fn=None):
    device = inputs.device
    # Compute gradients (but don't apply them)
    net.zero_grad()


    try:
        conds = get_sptk_n(inputs, targets, net, device)
    except Exception as e:
        conds= np.nan

    return conds
