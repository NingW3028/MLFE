import torch
import numpy as np
import time

def compute_epsinas(net, inputs, targets, weights=[1e-3, 1e-1]):
    """
    epsinas: Evaluate network using two constant weight initializations.
    
    Core idea:
    1. Initialize network with two different constant weights
    2. Compute normalized difference between two outputs
    3. MAE/Mean as the score
    
    Args:
        net: Neural network
        inputs: Input data
        targets: Target labels (unused)
        weights: Two initialization weights [w1, w2]
        seed: Random seed
    
    Returns:
        (score, time): epsinas score and computation time
    """
    start_time = time.time()
    
    device = next(net.parameters()).device
    inputs = inputs.to(device)
    
    preds = []
    
    for weight in weights:
        
        # Initialize all weights with constant value
        def initialize_weights(m):
            fill_bias = False
            if hasattr(m, 'bias') and m.bias is not None:
                fill_bias = True
            if fill_bias:
                torch.nn.init.constant_(m.bias, 0)
            
            fill_weight = False
            if hasattr(m, 'weight'):
                fill_weight = True
            if hasattr(m, 'affine') and not m.affine:
                fill_weight = False
            
            if fill_weight:
                torch.nn.init.constant_(m.weight, weight)
        
        net.apply(initialize_weights)
        
        # Forward pass
        net.eval()
        with torch.no_grad():
            output = net(inputs)
            if isinstance(output, tuple):
                output = output[0]
        
        # Normalize output
        pred = output.cpu().detach().numpy().flatten()
        pred_min = np.nanmin(pred)
        pred_max = np.nanmax(pred)
        if pred_max > pred_min:
            pred_norm = (pred - pred_min) / (pred_max - pred_min)
        else:
            pred_norm = pred
        preds.append(pred_norm)
    
    # Compute score
    preds = np.array(preds)
    # Do not filter zeros, as small weight init may produce zero outputs
    
    # Check for valid data
    if np.all(np.isnan(preds)):
        return 0.0, time.time() - start_time
    
    mae = np.abs(preds[0, :] - preds[1, :])
    mae_mean = np.nanmean(mae)
    preds_mean = np.nanmean(preds)
    if np.isnan(mae_mean) or np.isnan(preds_mean) or preds_mean == 0:
        score = 0.0
    else:
        score = mae_mean / (preds_mean + 1e-10)
    
    elapsed_time = time.time() - start_time
    return score, elapsed_time
