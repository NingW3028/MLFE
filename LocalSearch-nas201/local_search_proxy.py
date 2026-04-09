import numpy as np
import copy
import random
from tqdm import tqdm
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import torch
from easydict import EasyDict as edict
from models import get_cell_based_tiny_net


class LocalSearchNAS201Proxy:
    # NAS-Bench-201 operations
    OPS_LIST = ['none', 'skip_connect', 'nor_conv_1x1', 'nor_conv_3x3', 'avg_pool_3x3']

    def __init__(self, proxy_name='synflow', max_evaluations=1000, dataset='cifar10',
                 device='cuda', debug=False):
        self.proxy_name = proxy_name
        self.max_evaluations = max_evaluations
        self.dataset = dataset
        self.evaluations = 0
        self.archive = []
        self.device = device
        self.debug = debug
        self.pbar = None

        self.num_ops = len(self.OPS_LIST)
        self.num_edges = 6  # 6 edges in NAS-201 cell

        # Number of classes per dataset
        self.num_classes = {'cifar10': 10, 'cifar100': 100, 'ImageNet16-120': 120}.get(dataset, 10)

    def genotype_to_arch_str(self, genotype):
        """Convert integer genotype [6 ints] to NAS-201 architecture string.
        
        Format: |op~0|+|op~0|op~1|+|op~0|op~1|op~2|
        6 edges: (0->1), (0->2), (1->2), (0->3), (1->3), (2->3)
        """
        ops = self.OPS_LIST
        arch_str = '|'
        # Node 1: edge from node 0
        arch_str += f'{ops[genotype[0]]}~0|+|'
        # Node 2: edges from node 0 and 1
        arch_str += f'{ops[genotype[1]]}~0|{ops[genotype[2]]}~1|+|'
        # Node 3: edges from node 0, 1, and 2
        arch_str += f'{ops[genotype[3]]}~0|{ops[genotype[4]]}~1|{ops[genotype[5]]}~2|'
        return arch_str

    def scalarize_fitness(self, proxy_score, params, weights):
        """Scalarize: weights[0]*proxy_score - weights[1]*normalized_params"""
        normalized_params = params / 10e6
        return weights[0] * proxy_score - weights[1] * normalized_params

    def evaluate(self, genotype, inputs, targets):
        """Evaluate architecture using proxy.
        
        Args:
            genotype: list of 6 integers, each in [0, num_ops)
            inputs: input tensor batch
            targets: target tensor batch
            
        Returns:
            (proxy_score, num_params) or (None, None) on failure
        """
        try:
            arch_str = self.genotype_to_arch_str(genotype)

            config = edict({
                'name': 'infer.tiny',
                'C': 16,
                'N': 5,
                'arch_str': arch_str,
                'num_classes': self.num_classes
            })

            net = get_cell_based_tiny_net(config)
            net = net.to(self.device)
            net.eval()

            # Compute proxy score based on proxy type
            if self.proxy_name == 'synflow':
                from pruners.measures.synflow import compute_synflow_per_weight
                score = compute_synflow_per_weight(net, inputs, targets, mode='param')
                if isinstance(score, tuple):
                    score = score[1]
                elif isinstance(score, list):
                    score = sum([s.sum().item() for s in score])
                elif torch.is_tensor(score):
                    score = score.sum().item()
            elif self.proxy_name == 'zen':
                from pruners.measures.zen import compute_zen
                score = compute_zen(net, inputs, targets)
            elif self.proxy_name == 'zico':
                from pruners.measures.zico import compute_zico
                try:
                    score = compute_zico(net, inputs, targets)
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
                score = compute_naswot(net, inputs, torch.float32, benchtype='tss', dataset=self.dataset)
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
                score = compute_MLFE(net, inputs, torch.float32, benchtype='tss', dataset=self.dataset)
            elif self.proxy_name == 'ES':
                from proxieslib.EntropyScore import compute_EntropyScore
                score = compute_EntropyScore(net, inputs, torch.float32, benchtype='tss', dataset=self.dataset)
            else:
                raise ValueError(f"Unknown proxy: {self.proxy_name}")

            if self.debug:
                print(f"[DEBUG] {self.proxy_name} raw score: {score}, type: {type(score)}")

            # Convert to float if needed
            if isinstance(score, tuple):
                score = float(score[0])
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
            net.zero_grad()
            del net
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            import gc
            gc.collect()
            return float(score), params
        except Exception as e:
            print(f"Error evaluating: {e}")
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
        """Perform local search from initial genotype"""
        current_genotype = genotype.copy()
        current_fitness = self.evaluate(current_genotype, inputs, targets)

        if current_fitness[0] is None:
            return None, None

        changed = True
        while changed and self.evaluations < self.max_evaluations:
            changed = False

            # Random order traversal
            indices = list(range(self.num_edges))
            random.shuffle(indices)

            for i in indices:
                if self.evaluations >= self.max_evaluations:
                    break

                # Try all operations at position i
                for op_idx in range(self.num_ops):
                    if op_idx == current_genotype[i]:
                        continue

                    neighbor = current_genotype.copy()
                    neighbor[i] = op_idx

                    new_fitness = self.evaluate(neighbor, inputs, targets)

                    if new_fitness[0] is not None and self.dominates(new_fitness, current_fitness, weights):
                        current_genotype = neighbor
                        current_fitness = new_fitness
                        changed = True
                        break

                if changed and not loop:
                    break

            if not loop:
                break

        return current_genotype, current_fitness

    def search(self, train_loader, num_searches=20):
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

            # Scalarization weights
            if i == 0:
                weights = [1.0, 0.0]
            elif i == 1:
                weights = [0.0, 1.0]
            else:
                weights = [i/(num_searches-1), 1-i/(num_searches-1)]

            # Generate random architecture
            genotype = self._generate_random_genotype()

            # Local search
            final_genotype, final_fitness = self.local_search(genotype, weights, inputs, targets)

            if final_fitness and final_fitness[0] is not None:
                arch_str = self.genotype_to_arch_str(final_genotype)
                results.append({
                    'genotype': final_genotype,
                    'arch_str': arch_str,
                    'proxy_score': final_fitness[0],
                    'params': final_fitness[1],
                    'weights': weights
                })
                print(f"Search {i+1}/{num_searches}: Score={final_fitness[0]:.2e}, "
                      f"Params={final_fitness[1]}, Arch={arch_str}")

        self.pbar = None
        pbar.close()
        return results

    def _generate_random_genotype(self):
        """Generate random NAS-201 genotype (list of 6 integers)"""
        return [random.randint(0, self.num_ops - 1) for _ in range(self.num_edges)]
