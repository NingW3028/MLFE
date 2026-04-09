import numpy as np
import copy
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from collections import namedtuple
from tqdm import tqdm

import torch
from xautodl.nas_infer_model.DXYs.CifarNet import NetworkCIFAR

Genotype = namedtuple('Genotype', 'normal normal_concat reduce reduce_concat')

class LocalSearchNAS301Proxy:
    def __init__(self, proxy_name='synflow', max_evaluations=1000, device='cuda', debug=False):
        self.proxy_name = proxy_name
        self.max_evaluations = max_evaluations
        self.evaluations = 0
        self.archive = []
        self.device = device
        self.debug = debug
        self.pbar = None  # Progress bar
        
        # DARTS operations (removed 'none')
        self.ops = ['max_pool_3x3', 'avg_pool_3x3', 'skip_connect', 
                    'sep_conv_3x3', 'sep_conv_5x5', 'dil_conv_3x3', 'dil_conv_5x5']
        self.num_ops = len(self.ops)
        self.num_edges = 14  # 4 intermediate nodes, each with edges from all previous
    
    def scalarize_fitness(self, proxy_score, params, weights):
        """Scalarize: weights[0]*proxy_score - weights[1]*normalized_params"""
        normalized_params = params / 10e6
        return weights[0] * proxy_score - weights[1] * normalized_params
    
    def genotype_to_list(self, genotype):
        """Convert genotype to flat list representation"""
        result = []
        for node_ops in genotype.normal:
            for op, inp in node_ops:
                result.append((self.ops.index(op), inp))
        for node_ops in genotype.reduce:
            for op, inp in node_ops:
                result.append((self.ops.index(op), inp))
        return result
    
    def list_to_genotype(self, gene_list):
        """Convert flat list to nested genotype"""
        def build_cell(ops_list):
            cell = []
            for i in range(0, len(ops_list), 2):
                op1_idx, inp1 = ops_list[i]
                op2_idx, inp2 = ops_list[i+1]
                cell.append(((self.ops[op1_idx], inp1), (self.ops[op2_idx], inp2)))
            return cell
        
        normal = build_cell(gene_list[:8])
        reduce = build_cell(gene_list[8:])
        return Genotype(normal=normal, normal_concat=[2,3,4,5], 
                       reduce=reduce, reduce_concat=[2,3,4,5])
    
    def evaluate(self, genotype, inputs, targets):
        """Evaluate architecture using proxy"""
        try:
            # Debug: print genotype structure
            # print(f"Normal: {genotype.normal}")
            # print(f"Reduce: {genotype.reduce}")
            
            net = NetworkCIFAR(C=16, N=2, stem_multiplier=3, auxiliary=False, 
                              genotype=genotype, num_classes=10)
            
            # Wrap to handle tuple output
            class NetWrapper(torch.nn.Module):
                def __init__(self, base_net):
                    super().__init__()
                    self.base_net = base_net
                def forward(self, x):
                    out = self.base_net(x)
                    return out[0] if isinstance(out, tuple) else out
            
            net = NetWrapper(net)
            net = net.to(self.device)
            net.eval()
            
            # Compute proxy score
            if self.proxy_name == 'synflow':
                from pruners.measures.synflow import compute_synflow_per_weight
                score = compute_synflow_per_weight(net, inputs, targets, mode='param')
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
                swap_scorer = SWAP(model=net, inputs=inputs, device=self.device, seed=42, regular=False)
                score = swap_scorer.forward()
                if score is not None:
                    score = score.item() if torch.is_tensor(score) else score
                else:
                    score = 0.0
            elif self.proxy_name == 'naswot':
                from proxieslib.naswot import compute_naswot
                score = compute_naswot(net, inputs, targets)
            elif self.proxy_name == 'epads':
                from proxieslib.epads import compute_epads
                score = compute_epads(net, inputs, targets)
            elif self.proxy_name == 'near':
                from proxieslib.near import compute_near
                score = compute_near(net, inputs, targets)
            elif self.proxy_name == 'dextr':
                from proxieslib.dextr import compute_dextr
                score = compute_dextr(net, inputs, targets)
            elif self.proxy_name == 'wrcor':
                from proxieslib.wrcor import compute_wrcor
                try:
                    score = compute_wrcor(net, inputs, targets)
                except Exception as e:
                    if self.debug:
                        print(f"[DEBUG] wrcor exception: {e}")
                    score = 0.0
            elif self.proxy_name == 'epsinas':
                from proxieslib.epsinas import compute_epsinas
                score = compute_epsinas(net, inputs, targets)
            elif self.proxy_name == 'MLFE':
                from proxieslib.MLFE import compute_MLFE
                score = compute_MLFE(net, inputs, torch.float32, benchtype='Darts', dataset='cifar10')
            elif self.proxy_name == 'ES':
                from proxieslib.EntropyScore import compute_EntropyScore
                score = compute_EntropyScore(net, inputs, torch.float32, benchtype='Darts', dataset='cifar10')
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
            import traceback
            if self.debug:
                print(f"[DEBUG] Error evaluating: {e}")
                traceback.print_exc()
            else:
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
    
    def local_search(self, genotype, weights, inputs, targets, loop=True):
        """Perform local search"""
        current_genotype = copy.deepcopy(genotype)
        current_fitness = self.evaluate(current_genotype, inputs, targets)
        
        if current_fitness[0] is None:
            return None, None
        
        changed = True
        iterations = 0
        while changed and self.evaluations < self.max_evaluations:
            changed = False
            iterations += 1
            
            # Try changing each operation
            gene_list = self.genotype_to_list(current_genotype)
            for i in range(len(gene_list)):
                op_idx, node = gene_list[i]
                for new_op_idx in range(self.num_ops):
                    if new_op_idx == op_idx:
                        continue
                    
                    new_gene_list = [g for g in gene_list]  # Deep copy
                    new_gene_list[i] = (new_op_idx, node)
                    new_genotype = self.list_to_genotype(new_gene_list)
                    new_fitness = self.evaluate(new_genotype, inputs, targets)
                    
                    if new_fitness[0] and self.dominates(new_fitness, current_fitness, weights):
                        current_genotype = new_genotype
                        current_fitness = new_fitness
                        gene_list = new_gene_list
                        changed = True
                        break
                if changed:
                    break
            
            if not loop or iterations > 5:  # Limit iterations
                break
        
        return current_genotype, current_fitness
    
    def search(self, train_loader, num_searches=10):
        """Run multiple local searches"""
        results = []
        
        # Get one batch of data
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
            
            # Try multiple times to get valid initial architecture
            max_retries = 5
            final_genotype = None
            final_fitness = None
            
            for retry in range(max_retries):
                genotype = self._generate_random_genotype()
                final_genotype, final_fitness = self.local_search(genotype, weights, inputs, targets)
                
                if final_fitness and final_fitness[0] is not None:
                    break
            
            if final_fitness and final_fitness[0] is not None:
                results.append({
                    'genotype': final_genotype,
                    'proxy_score': final_fitness[0],
                    'params': final_fitness[1],
                    'weights': weights
                })
                if self.debug:
                    print(f"[DEBUG] Search {i+1}/{num_searches}: Score={final_fitness[0]:.2e}, Params={final_fitness[1]}, Evals={self.evaluations}")
                else:
                    print(f"Search {i+1}/{num_searches}: Score={final_fitness[0]:.2e}, Params={final_fitness[1]}")
            else:
                print(f"Search {i+1}/{num_searches}: Failed to find valid architecture")
        
        self.pbar = None
        pbar.close()
        return results
    
    def _generate_random_genotype(self):
        """Generate random DARTS genotype - correct nested format"""
        # Format: [((op1, input1), (op2, input2)), ...] for 4 nodes
        def random_cell():
            cell = []
            for i in range(4):  # 4 intermediate nodes
                inputs = np.random.choice(range(i+2), 2, replace=False).tolist()
                ops = [np.random.choice(self.ops) for _ in range(2)]
                cell.append(((ops[0], inputs[0]), (ops[1], inputs[1])))
            return cell
        
        normal = random_cell()
        reduce = random_cell()
        
        return Genotype(normal=normal, normal_concat=[2,3,4,5], 
                       reduce=reduce, reduce_concat=[2,3,4,5])
