import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from local_search_proxy import LocalSearchNAS101Proxy
import json
import time
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--proxy', type=str, default='MLFE')
parser.add_argument('--max_evals', type=int, default=100)
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
ls = LocalSearchNAS101Proxy(proxy_name=proxy_name, max_evaluations=max_evals, device=device, debug=True)

print(f"Starting Local Search with {proxy_name} proxy...")
start_time = time.time()

# Run search
results = ls.search(train_loader, num_searches=10)

elapsed = time.time() - start_time
print(f"\nSearch completed in {elapsed:.2f}s")
print(f"Total evaluations: {ls.evaluations}")
print(f"Found {len(results)} architectures")

output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results', 'LocalSearch-nas101')
os.makedirs(output_dir, exist_ok=True)

# Query ground-truth accuracy from nas101_cache.pkl
cache_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'nas101_cache.pkl')
if os.path.exists(cache_path):
    import pickle
    print(f"\nQuerying ground-truth accuracy from nas101_cache.pkl...")
    with open(cache_path, 'rb') as f:
        nb101_cache = pickle.load(f)
    for r in results:
        matrix_tuple = tuple(tuple(row) for row in r['matrix'])
        ops_tuple = tuple(r['ops'])
        key = (matrix_tuple, ops_tuple)
        if key in nb101_cache:
            acc = nb101_cache[key]['accuracy']
            r['test_accuracy'] = float(acc)
            print(f"  Ops: {r['ops'][1:-1]}  ->  Test Acc: {acc*100:.2f}%")
        else:
            r['test_accuracy'] = None
            print(f"  Ops: {r['ops'][1:-1]}  ->  Not found in cache")
    del nb101_cache
else:
    print(f"\nnas101_cache.pkl not found, skipping accuracy query.")

# Save results
output = {
    'proxy': proxy_name,
    'results': results,
    'total_evaluations': ls.evaluations,
    'time': elapsed
}

output_file = os.path.join(output_dir, f'ls_results_{proxy_name}.json')

with open(output_file, 'w') as f:
    json.dump(output, f, indent=2, default=lambda x: x.tolist() if hasattr(x, 'tolist') else x)

print(f"\nResults saved to {output_file}")
