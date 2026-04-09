import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
# Cross-environment import for EvoxBench (Python 3.7 env)
if os.environ.get('EVOXBENCH_SITE_PACKAGES'):
    sys.path.append(os.environ['EVOXBENCH_SITE_PACKAGES'])

import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from local_search_proxy import LocalSearchNAS301Proxy
import json
import time
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--proxy', type=str, default='MLFE')
parser.add_argument('--max_evals', type=int, default=100)
parser.add_argument('--bench', type=str, default=None, choices=['nas301', 'evoxbench'],
                    help='Query accuracy after search: nas301 (test acc, deterministic) or evoxbench (valid acc, with noise)')
args = parser.parse_args()

# Setup
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# Load CIFAR-10 data
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
# NOTE: Place CIFAR-10 data in 'data/cifar10' directory
trainset = datasets.CIFAR10(root='data/cifar10', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True, num_workers=0, drop_last=True)

# Proxy to use
proxy_name = args.proxy
print(f"Using proxy: {proxy_name}")

# Initialize Local Search
max_evals = args.max_evals
ls = LocalSearchNAS301Proxy(proxy_name=proxy_name, max_evaluations=max_evals, device=device, debug=True)

print(f"Starting Local Search with {proxy_name} proxy...")
start_time = time.time()

# Run search
results = ls.search(train_loader, num_searches=10)

elapsed = time.time() - start_time
print(f"\nSearch completed in {elapsed:.2f}s")
print(f"Total evaluations: {ls.evaluations}")
print(f"Found {len(results)} architectures")

# Query ground-truth accuracy if benchmark specified
if args.bench:
    true_eval = (args.bench == 'nas301')
    acc_label = 'Test Acc' if true_eval else 'Valid Acc'
    print(f"\nQuerying {args.bench} accuracy (true_eval={true_eval})...")
    try:
        from evoxbench.benchmarks import DARTSSearchSpace, DARTSBenchmark
        from evoxbench.benchmarks.darts import Genotype as EvoxGenotype
        ss = DARTSSearchSpace()
        benchmark = DARTSBenchmark(objs='err&params', normalized_objectives=False)

        def convert_genotype(geno):
            """Convert nested genotype ((op,in),(op,in)) to flat EvoxBench format [(op,in),...]"""
            def flatten_cell(cell):
                flat = []
                for node in cell:
                    for edge in node:
                        flat.append(edge)
                return flat
            return EvoxGenotype(
                normal=flatten_cell(geno.normal),
                normal_concat=list(geno.normal_concat),
                reduce=flatten_cell(geno.reduce),
                reduce_concat=list(geno.reduce_concat)
            )

        for r in results:
            try:
                geno = r['genotype']
                evox_geno = convert_genotype(geno)
                encoded = ss.encode([evox_geno])
                res = benchmark.evaluate(encoded, true_eval=true_eval)
                acc = (1.0 - res[0][0]) * 100.0
                r['test_accuracy'] = float(acc)
                r['bench'] = args.bench
                print(f"  Genotype -> {acc_label}: {acc:.2f}%")
            except Exception as e:
                r['test_accuracy'] = None
                print(f"  Genotype -> Error: {e}")
    except ImportError:
        print("  evoxbench not available. Set EVOXBENCH_SITE_PACKAGES env var.")

# Save results
output = {
    'proxy': proxy_name,
    'bench': args.bench,
    'results': [{**r, 'genotype': str(r['genotype'])} for r in results],
    'total_evaluations': ls.evaluations,
    'time': elapsed
}

output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results', 'LocalSearch-Darts')
os.makedirs(output_dir, exist_ok=True)
output_file = os.path.join(output_dir, f'ls_results_{proxy_name}.json')

with open(output_file, 'w') as f:
    json.dump(output, f, indent=2)

print(f"\nResults saved to {output_file}")
