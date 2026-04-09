import copy
import time
import re
from scipy.stats import skew
from scipy.stats import kurtosis
import random
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F



def discrete_entropy_onepass(tensor_arroy):
    # tensor_arroy: [batch, N]
    batch_size, N = tensor_arroy.shape
    B = N
    min_val = tensor_arroy.min(dim=1, keepdim=True).values
    max_val = tensor_arroy.max(dim=1, keepdim=True).values
    val_range = max_val - min_val
    # Handle constant rows
    constant_mask = (val_range.squeeze(1) < 1e-10)
    val_range = val_range.clamp(min=1e-10)
    # Partition [min, max] into B equal-width bins, assign each entry to nearest bin
    bin_indices = ((tensor_arroy - min_val) / val_range * (B - 1)).long().clamp(0, B - 1)
    # Batch bincount via scatter
    counts = torch.zeros(batch_size, B, device=tensor_arroy.device)
    counts.scatter_add_(1, bin_indices, torch.ones_like(tensor_arroy))
    # Compute entropy: mask out zero-count bins
    mask = counts > 0
    probabilities = counts / counts.sum(dim=1, keepdim=True)
    log_prob = torch.zeros_like(probabilities)
    log_prob[mask] = torch.log2(probabilities[mask])
    entropy = -(probabilities * log_prob).sum(dim=1)
    entropy[constant_mask] = 0.0
    return entropy.tolist()


def continue_entropy_onepass(tensor_arroy):
    probabilities = torch.softmax(torch.abs(tensor_arroy), dim=1)
    entropy = -torch.sum(probabilities * torch.log2(probabilities),dim=1)
    return entropy.tolist()

def var_onepass(tensor_arroy):
    var_list=[]
    for i in range(tensor_arroy.shape[0]):
        var_list.append(torch.var(tensor_arroy[i]).item())
    return var_list


def get_matrix_for_entropy(feature_map,conv_size,sqrt_cell_num):
    feature_map_shape=feature_map.shape
    # Support both 2D (T, D) from transformers and 3D (C, H, W) from CNNs
    if len(feature_map_shape)==2:
        h, w = feature_map_shape[0], feature_map_shape[1]
        slice_fn = lambda fm, i, j: fm[i:i+conv_size,j:j+conv_size]
    else:
        h, w = feature_map_shape[1], feature_map_shape[2]
        slice_fn = lambda fm, i, j: fm[:,i:i+conv_size,j:j+conv_size]
    i_list=[i+int(conv_size/2) for i in range(int(h)-2*int(conv_size/2)-conv_size+1)]
    j_list=[j+int(conv_size/2) for j in range(int(w)-2*int(conv_size/2)-conv_size+1)]
    if len(i_list)==0 or len(j_list)==0:
        i_list=[i for i in range(int(h)-conv_size+1)]
        j_list=[j for j in range(int(w)-conv_size+1)]
    if sqrt_cell_num != 0 and len(i_list) > 1 and len(j_list) > 1:
        i_list_indexes = np.linspace(0, len(i_list)-1, sqrt_cell_num, endpoint=True).astype(int).tolist()
        j_list_indexes = np.linspace(0, len(j_list)-1, sqrt_cell_num, endpoint=True).astype(int).tolist()
        i_list = [i_list[i] for i in i_list_indexes]
        j_list = [j_list[j] for j in j_list_indexes]
    matrix_list=[]
    feature_map_list=[]
    matrix_list2=[]
    
    for i in i_list:
        for j in j_list:
            matrix_list.append(slice_fn(feature_map,i,j).reshape(1,-1))
            feature_map_list.append(slice_fn(feature_map,i,j))
    matrix=torch.cat(matrix_list,dim=0)
    discrete_entropy_list=discrete_entropy_onepass(matrix)
    continue_entropy_list=continue_entropy_onepass(matrix)
    var_list=var_onepass(matrix)
    
    norm_list=[]
    for i in range(len(matrix_list)):
        norm_list.append(torch.norm(feature_map_list[i]).item())
    return discrete_entropy_list,continue_entropy_list,norm_list,var_list




def get_score(net, x, device, dtype,split_data,benchtype='tss',dataset='cifar10',conv_mode='fixed'):
    result_list = []

    def forward_hook(module, data_input, data_output):
        fea_patch_list=[]
        if isinstance(data_output, torch.Tensor):
            fea = data_output[0].detach()
        elif isinstance(data_output, tuple) and len(data_output) > 0:
            fea = data_output[0][0].detach() if isinstance(data_output[0], torch.Tensor) else data_output[0].detach()
        else:
            return
        fea[torch.isnan(fea)] = 0
        fea[torch.isinf(fea)] = 0
        sqrt_cell_num=4
        if len(fea.shape)<2:
            return
        fea_cpu = fea.cpu().detach()
        if conv_mode == 'auto':
            # Auto-detect conv_size from layer's kernel size
            conv_size=3
            for sub in module.modules():
                if isinstance(sub, nn.Conv2d):
                    k = max(sub.kernel_size) if isinstance(sub.kernel_size, tuple) else sub.kernel_size
                    if k > conv_size:
                        conv_size = k
            if len(fea.shape) == 3 and fea.shape[1] < conv_size:
                conv_size = fea.shape[1]
            if len(fea.shape) == 3 and fea.shape[1] >= conv_size:
                entropy_list,continue_entropy_list,_,_=get_matrix_for_entropy(fea_cpu,conv_size,sqrt_cell_num)
                for i in range(len(entropy_list)):
                    fea_patch_list.append(entropy_list[i]*continue_entropy_list[i])
        else:
            # Multi-patch mode: iterate over multiple conv sizes
            multi_conv_sizes=[3]
            for conv_size in multi_conv_sizes:
                if len(fea.shape) == 3 and (fea.shape[1] < conv_size or fea.shape[2] < conv_size):
                    continue
                if len(fea.shape) == 2 and (fea.shape[0] < conv_size or fea.shape[1] < conv_size):
                    continue
                entropy_list,continue_entropy_list,_,_=get_matrix_for_entropy(fea_cpu,conv_size,sqrt_cell_num)
                for i in range(len(entropy_list)):
                    fea_patch_list.append(entropy_list[i]*continue_entropy_list[i])

        fea_patch_list=np.array(fea_patch_list)
        fea_patch_list[np.isnan(fea_patch_list)]=0
        fea_patch_list[np.isinf(fea_patch_list)]=0
        non_zero_mask = fea_patch_list != 0
        non_zero_elements = fea_patch_list[non_zero_mask]
        if len(non_zero_elements)>0:
            result_list.append(np.sum(non_zero_elements))
        else:
            result_list.append(0)

    def forward_hook_transformer(module, data_input, data_output):
        fea_patch_list=[]
        if isinstance(data_output, torch.Tensor):
            fea = data_output[0].detach()  # (T, D)
        elif isinstance(data_output, tuple) and len(data_output) > 0:
            fea = data_output[0][0].detach() if isinstance(data_output[0], torch.Tensor) else data_output[0].detach()
        else:
            return
        fea[torch.isnan(fea)] = 0
        fea[torch.isinf(fea)] = 0
        if len(fea.shape) != 2:
            return
        T, D = fea.shape
        # Uniformly sample tokens from T
        num_sample_tokens = min(16, T)
        token_indices = np.linspace(0, T - 1, num_sample_tokens, endpoint=True).astype(int)
        # Each sampled token is a 1D vector of dimension D
        sampled = fea[token_indices].cpu().detach()  # (num_sample_tokens, D)
        # Compute entropy per token embedding (each row is a D-dim vector)
        discrete_entropy_list = discrete_entropy_onepass(sampled)
        continue_entropy_list = continue_entropy_onepass(sampled)
        for i in range(len(discrete_entropy_list)):
            fea_patch_list.append(discrete_entropy_list[i] * continue_entropy_list[i])

        fea_patch_list = np.array(fea_patch_list)
        fea_patch_list[np.isnan(fea_patch_list)] = 0
        fea_patch_list[np.isinf(fea_patch_list)] = 0
        non_zero_mask = fea_patch_list != 0
        non_zero_elements = fea_patch_list[non_zero_mask]
        if len(non_zero_elements) > 0:
            result_list.append(np.sum(non_zero_elements))
        else:
            result_list.append(0)
        

    for name, modules in net.named_modules():
        if benchtype=='tss':
            if ('cells.16' in name) or ('cells.15' in name)or ('cells.14' in name):
                modules.register_forward_hook(forward_hook)
            elif ( ('layers'in name)  ) and 'cells' not in name:
                modules.register_forward_hook(forward_hook)
        elif benchtype=='Darts':
            if ('cells.7' in name) or ('cells.6' in name)or ('cells.5' in name):
                modules.register_forward_hook(forward_hook)
        elif benchtype=='AutoFormer':
            if re.match(r'^blocks\.\d+$', name):
                modules.register_forward_hook(forward_hook_transformer)


    N = x.shape[0]
    for sp in range(split_data):
        st = sp * N // split_data
        en = (sp + 1) * N // split_data
        one_x=torch.ones_like(x[st:en]).to(device)
        one_x=torch.tensor(one_x,dtype= dtype)
        y = net(one_x)
        
        
    results = torch.tensor(result_list)
    results[torch.isnan(results)]=0
    results[torch.isinf(results)]=0
    
    results = results[torch.logical_not(torch.isnan(results))]
 
    len_results=len(results)
    non_zero_mask = results != 0
    non_zero_elements = results[non_zero_mask]
    non_zero_elements,_=torch.sort(non_zero_elements)  # sort the non_zero_elements (sort, idnex)
    if len(non_zero_elements)>0:
        len_non_zero_elements=len(non_zero_elements)
        diff_list=[]
        for i in range(len_non_zero_elements):
            diff=(non_zero_elements[(len_non_zero_elements-1)-(i)]-non_zero_elements[i]).item()
            if (len_non_zero_elements-1-i)<i:
                diff_list=torch.tensor(diff_list)
                break
            diff_list.append(diff)
        diff_list=torch.tensor(diff_list)
        v = np.log(torch.max(diff_list))*torch.sum((results))  # L_A is a constant across architectures, thus omitted
        
    else:
        v = torch.tensor(0,dtype=dtype)
 
    
    result_list.clear()
    return v.item()



def compute_MLFE(net, inputs, dtype,split_data=1, loss_fn=None,benchtype='tss',dataset='cifar10',conv_mode='fixed'):
    device = inputs.device
    net.zero_grad()
    MultiSacleLE = get_score(net, inputs, device, dtype,split_data=split_data,benchtype=benchtype,dataset=dataset,conv_mode=conv_mode)
    return MultiSacleLE
