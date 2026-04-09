"""
NAS-301 Local Search with Accuracy Query
Uses EvoxBench DARTSBenchmark API to predict architecture accuracy.

--bench nas301:   test accuracy predictor (deterministic)
--bench evoxbench: validation accuracy predictor (with noise)
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
# If evoxbench is in a separate environment, set EVOXBENCH_SITE_PACKAGES
if os.environ.get('EVOXBENCH_SITE_PACKAGES'):
    sys.path.append(os.environ['EVOXBENCH_SITE_PACKAGES'])
import argparse
import pickle
import random
import numpy as np
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from local_search_proxy import LocalSearchNAS301Proxy


def load_evoxbench():
    """Load EvoxBench DARTSBenchmark for accuracy evaluation"""
    try:
        from evoxbench.benchmarks import DARTSSearchSpace, DARTSBenchmark
        ss = DARTSSearchSpace()
        benchmark = DARTSBenchmark(objs='err&params', normalized_objectives=False)
        return ss, benchmark
    except Exception as e:
        print(f"Warning: Failed to load EvoxBench: {e}")
        print("Accuracy query will be disabled")
        return None, None


def flatten_genotype(genotype):
    """Convert nested genotype to flat DARTS format for EvoxBench.
    
    Nested: normal=[((op1,in1),(op2,in2)), ((op3,in3),(op4,in4)), ...]
    Flat:   normal=[(op1,in1),(op2,in2),(op3,in3),(op4,in4), ...]
    """
    from collections import namedtuple
    Genotype = namedtuple('Genotype', 'normal normal_concat reduce reduce_concat')
    
    def flatten_cell(cell):
        flat = []
        for node in cell:
            if isinstance(node[0], tuple):
                # Nested format: ((op, in), (op, in))
                for op_inp in node:
                    flat.append(op_inp)
            else:
                # Already flat: (op, in)
                flat.append(node)
        return flat
    
    return Genotype(
        normal=flatten_cell(genotype.normal),
        normal_concat=genotype.normal_concat,
        reduce=flatten_cell(genotype.reduce),
        reduce_concat=genotype.reduce_concat
    )


def query_accuracy_from_evoxbench(genotype, ss, benchmark, true_eval=True):
    """Query architecture accuracy from EvoxBench DARTSBenchmark"""
    if ss is None or benchmark is None:
        return None
    try:
        flat_geno = flatten_genotype(genotype)
        encoded = ss.encode([flat_geno])
        results = benchmark.evaluate(encoded, true_eval=true_eval)  # [err, params]
        acc = (1.0 - results[0][0]) * 100.0  # err -> acc%
        return acc
    except Exception as e:
        print(f"Error querying accuracy: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description='MacroNAS Local Search for NAS-301 / EvoxBench-DARTS')
    parser.add_argument('--proxy', type=str, required=True)
    parser.add_argument('--bench', type=str, default='nas301', choices=['nas301', 'evoxbench'],
                        help='Benchmark: nas301 (test acc, deterministic) or evoxbench (valid acc, with noise)')
    parser.add_argument('--max_evals', type=int, default=500)
    parser.add_argument('--num_searches', type=int, default=10)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--data_root', type=str, default='./data',
                        help='Root directory for CIFAR-10 (auto-downloaded if absent)')
    args = parser.parse_args()
    
    # Set seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    true_eval = (args.bench == 'nas301')
    print(f"Device: {device}")
    print(f"Proxy: {args.proxy}")
    print(f"Bench: {args.bench} (true_eval={true_eval})")
    print(f"Max evals: {args.max_evals}")
    print(f"Seed: {args.seed}")
    
    # Load EvoxBench for accuracy evaluation
    print("Loading EvoxBench DARTSBenchmark...")
    ss, benchmark = load_evoxbench()
    
    # Load data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    trainset = datasets.CIFAR10(root=os.path.join(args.data_root, 'cifar10'),
                                train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True, num_workers=0, drop_last=True)
    
    # Run search
    ls = LocalSearchNAS301Proxy(proxy_name=args.proxy, max_evaluations=args.max_evals, device=device, debug=False)
    results = ls.search(train_loader, num_searches=args.num_searches)
    
    # Query accuracies via EvoxBench
    if ss is not None and benchmark is not None:
        print(f"Querying accuracies from EvoxBench ({args.bench})...")
        for res in results:
            acc = query_accuracy_from_evoxbench(res['genotype'], ss, benchmark, true_eval=true_eval)
            res['predicted_accuracy'] = acc
            if acc is not None:
                print(f"  Proxy: {res['proxy_score']:.2e}, Predicted Acc: {acc:.2f}%")
    
    # Find best by proxy score
    if results:
        best = max(results, key=lambda x: x['proxy_score'])
        output_data = {
            'best_genotype': str(best['genotype']),
            'best_proxy_score': best['proxy_score'],
            'best_predicted_accuracy': best.get('predicted_accuracy'),
            'best_params': best['params'],
            'all_results': results,
            'proxy': args.proxy,
            'bench': args.bench,
            'seed': args.seed
        }
    else:
        output_data = {'error': 'No results', 'proxy': args.proxy}
    
    # Save
    with open(args.output, 'wb') as f:
        pickle.dump(output_data, f)
    
    print(f"Results saved to {args.output}")


if __name__ == '__main__':
    main()
