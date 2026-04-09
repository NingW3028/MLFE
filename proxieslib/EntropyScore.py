import copy
import time
from scipy.stats import skew
from scipy.stats import kurtosis
import random
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np


def get_score(net, x, device, dtype,split_data,benchtype='tss',dataset='cifar10'):
    result_list = []
    hook_handles = []

    def forward_hook(module, data_input, data_output):
        fea_patch_list=[]
        channel_entropy_list=[]
        fea = data_output[0].detach()
        fea[torch.isnan(fea)] = 0
        fea[torch.isinf(fea)] = 0
        
        featur_map=fea.cpu().detach()
        featur_map=torch.reshape(featur_map,(1,-1))
        ES=(torch.mean(featur_map*torch.log(featur_map+1e-3))).item()
        #print('ES',ES)
        result_list.append(ES)   
        
    for name, modules in net.named_modules():
        #print(name)
        if benchtype=='tss':
           # if 'cells.16' in name:
            #if ('cells.16' in name) or ('cells.15' in name)or ('cells.14' in name):
            if ('cells.16' in name):
                #print(name)
                modules.register_forward_hook(forward_hook)
            elif ( ('layers'in name)  ) and 'cells' not in name:
            #elif ( ('layers' in name) ) and 'cells' not in name:
                modules.register_forward_hook(forward_hook)
        elif benchtype=='sss':
            if 'cells' in name:
                modules.register_forward_hook(forward_hook)
        elif benchtype=='Darts':
            #if ('cells.7' in name) or ('cells.6' in name):
            #if ('cells.7' in name):
            #if ('cells.7' in name) or ('cells.6' in name):
            if ('cells' in name):
                modules.register_forward_hook(forward_hook)


    N = x.shape[0]
    for sp in range(split_data):
        st = sp * N // split_data
        en = (sp + 1) * N // split_data
        one_x=torch.ones_like(x[st:en]).to(device)
        random_x=torch.rand_like(x[st:en]).to(device)
        one_x=torch.tensor(one_x,dtype= dtype)
        #print('random_x_shape',random_x.shape)
        y = net(random_x)
        
    results = torch.tensor(result_list)
    results[torch.isnan(results)]=0
    results[torch.isinf(results)]=0
    
    results = results[torch.logical_not(torch.isnan(results))]
    #print('results',results)
    v = torch.sum((results))
 
    
    result_list.clear()
    #print('MultiSacleLE',v)
    return v.item()



def compute_EntropyScore(net, inputs, dtype,split_data=1, loss_fn=None,benchtype='tss',dataset='cifar10'):
    device = inputs.device
    net.zero_grad()
    #try:
    EntropyScore = get_score(net, inputs, device, dtype,split_data=split_data,benchtype=benchtype,dataset=dataset)
    #except  Exception as e:
    #    print(e)
    #    MultiSacleLE=np.nan
    return EntropyScore
