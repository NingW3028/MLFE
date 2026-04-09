import numpy as np
import copy
from tqdm import tqdm
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import torch
from nasbench_pytorch.model import Network as NBNetwork

class LocalSearchNAS101Proxy:
    def __init__(self, proxy_name='synflow', max_evaluations=1000, device='cuda', debug=False):
        self.proxy_name = proxy_name
        self.max_evaluations = max_evaluations
        self.evaluations = 0
        self.archive = []
        self.device = device
        self.debug = debug
        self.pbar = None  # Progress bar
    
    def scalarize_fitness(self, proxy_score, params, weights):
        """Scalarize: weights[0]*proxy_score - weights[1]*normalized_params"""
        normalized_params = params / 10e6
        return weights[0] * proxy_score - weights[1] * normalized_params
    
    def evaluate(self, matrix, ops, inputs, targets):
        """Evaluate architecture using proxy"""
        try:
            net = NBNetwork((matrix, ops))
            net = net.to(self.device)
            net.eval()
            
            # Compute proxy score based on proxy type
            if self.proxy_name == 'synflow':
                from pruners.measures.synflow import compute_synflow_per_weight
                score = compute_synflow_per_weight(net, inputs, targets, mode='param')
                # synflow returns list of tensors, sum them
                if isinstance(score, list):
                    score = sum([s.sum().item() for s in score])
                elif torch.is_tensor(score):
                    score = score.sum().item()
            elif self.proxy_name == 'zen':
                from pruners.measures.zen import compute_zen
                score = compute_zen(net, inputs, targets)
            elif self.proxy_name == 'zico':
                from pruners.measures.zico import compute_zico
                import torch.nn as nn
                loss_fn = nn.CrossEntropyLoss()
                try:
                    score = compute_zico(net, inputs, targets, loss_fn=loss_fn)
                except Exception as e:
                    if self.debug:
                        print(f"[DEBUG] zico exception: {e}")
                    score = 0.0
            elif self.proxy_name == 'meco':
                from pruners.measures.meco import compute_meco
                score = compute_meco(net, inputs, targets)
            elif self.proxy_name == 'swap':
                from proxieslib.swap import SWAP
                swap_model = SWAP(model=net, inputs=inputs, device=self.device)
                score = swap_model.forward()
            elif self.proxy_name == 'naswot':
                from proxieslib.naswot import compute_naswot
                score = compute_naswot(net, inputs, torch.float32, benchtype='101', dataset='cifar10')
            elif self.proxy_name == 'wrcor':
                from proxieslib.wrcor import compute_wrcor
                score = compute_wrcor(net, inputs, targets)
            elif self.proxy_name == 'epads':
                from proxieslib.epads import compute_epads
                score = compute_epads(net, inputs, targets)
            elif self.proxy_name == 'near':
                from proxieslib.near import compute_near
                score = compute_near(net, inputs, targets)
            elif self.proxy_name == 'dextr':
                from proxieslib.dextr import compute_dextr
                score = compute_dextr(net, inputs, targets)
            elif self.proxy_name == 'epsinas':
                from proxieslib.epsinas import compute_epsinas
                score = compute_epsinas(net, inputs, targets)
            elif self.proxy_name == 'MLFE':
                from proxieslib.MLFE import compute_MLFE
                # Use only first sample to reduce memory
                score = compute_MLFE(net, inputs[:1], torch.float32, benchtype='tss', dataset='cifar10')
            elif self.proxy_name == 'ES':
                from proxieslib.EntropyScore import compute_EntropyScore
                score = compute_EntropyScore(net, inputs, torch.float32, benchtype='tss', dataset='cifar10')
            else:
                raise ValueError(f"Unknown proxy: {self.proxy_name}")
            
            if self.debug:
                print(f"[DEBUG] {self.proxy_name} raw score: {score}, type: {type(score)}")
                if isinstance(score, np.ndarray):
                    print(f"[DEBUG] Array shape: {score.shape}, size: {score.size}")
            
            # Convert to float if needed
            if isinstance(score, tuple):
                score = float(score[0])  # Take first element of tuple
            elif torch.is_tensor(score):
                if score.numel() == 0:
                    score = 0.0
                else:
                    score = score.cpu().item() if score.numel() == 1 else score.sum().cpu().item()
            elif isinstance(score, np.ndarray):
                if score.size == 0:
                    score = 0.0
                else:
                    score = float(score.item()) if score.size == 1 else float(score.sum())
            
            # Handle invalid scores
            if score is None or np.isnan(score) or np.isinf(score):
                score = 0.0
            
            params = sum(p.numel() for p in net.parameters())
            
            self.evaluations += 1
            if self.pbar:
                self.pbar.update(1)
            
            # Clean up
            net.zero_grad()  # Clear gradients
            del net
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            import gc
            gc.collect()
            return float(score), params
        except Exception as e:
            print(f"Error evaluating: {e}")
            # Clean up on error
            try:
                del net
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                import gc
                gc.collect()
            except:
                pass
            return None, None
    
    def dominates(self, fitness1, fitness2, weights):
        """Check if fitness1 dominates fitness2"""
        score1 = self.scalarize_fitness(fitness1[0], fitness1[1], weights)
        score2 = self.scalarize_fitness(fitness2[0], fitness2[1], weights)
        return score1 > score2
    
    def local_search(self, matrix, ops, weights, inputs, targets, loop=True):
        """Perform local search"""
        current_matrix = copy.deepcopy(matrix)
        current_ops = copy.deepcopy(ops)
        current_fitness = self.evaluate(current_matrix, current_ops, inputs, targets)
        
        if current_fitness[0] is None:
            return None, None, None
        
        changed = True
        while changed and self.evaluations < self.max_evaluations:
            changed = False
            
            # Try changing operations
            ops_list = ['conv3x3-bn-relu', 'conv1x1-bn-relu', 'maxpool3x3']
            for i in range(1, len(current_ops)-1):
                for op in ops_list:
                    if op == current_ops[i]:
                        continue
                    
                    new_ops = copy.deepcopy(current_ops)
                    new_ops[i] = op
                    new_fitness = self.evaluate(current_matrix, new_ops, inputs, targets)
                    
                    if new_fitness[0] and self.dominates(new_fitness, current_fitness, weights):
                        current_ops = new_ops
                        current_fitness = new_fitness
                        changed = True
            
            if not loop:
                break
        
        return current_matrix, current_ops, current_fitness
    
    def search(self, train_loader, num_searches=10):
        """Run multiple local searches"""
        results = []
        
        # Get one batch of data for all evaluations
        data_iter = iter(train_loader)
        inputs, targets = next(data_iter)
        inputs = inputs.to(self.device)
        targets = targets.to(self.device)
        
        pbar = tqdm(total=self.max_evaluations, desc="Evaluations", unit="eval")
        self.pbar = pbar
        
        for i in range(num_searches):
            if self.evaluations >= self.max_evaluations:
                break
            
            prev_evals = self.evaluations
            
            # Scalarization weights
            if i == 0:
                weights = [1.0, 0.0]
            elif i == 1:
                weights = [0.0, 1.0]
            else:
                weights = [i/(num_searches-1), 1-i/(num_searches-1)]
            
            # Generate valid architecture
            matrix, ops = self._generate_valid_arch()
            
            # Local search
            final_matrix, final_ops, final_fitness = self.local_search(matrix, ops, weights, inputs, targets)
            
            if final_fitness and final_fitness[0] is not None:
                results.append({
                    'matrix': final_matrix.tolist() if hasattr(final_matrix, 'tolist') else final_matrix,
                    'ops': final_ops,
                    'proxy_score': final_fitness[0],
                    'params': final_fitness[1],
                    'weights': weights
                })
                print(f"Search {i+1}/{num_searches}: Score={final_fitness[0]:.2f}, Params={final_fitness[1]}")
        
        self.pbar = None
        pbar.close()
        return results
    
    def _generate_valid_arch(self):
        """Generate valid architecture"""
        matrix = np.zeros((7, 7), dtype=int)
        for i in range(6):
            matrix[i, i+1] = 1
        
        ops = ['input'] + [np.random.choice(['conv3x3-bn-relu', 'conv1x1-bn-relu', 'maxpool3x3']) 
                           for _ in range(5)] + ['output']
        
        return matrix, ops

