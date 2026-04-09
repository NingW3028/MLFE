import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from local_search_proxy import LocalSearchNAS201Proxy
import json
import time
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--proxy', type=str, default='MLFE')
parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10', 'cifar100', 'ImageNet16-120'])
parser.add_argument('--max_evals', type=int, default=100)
parser.add_argument('--api_path', type=str, default=None, help='Path to NAS-Bench-201 API file (.pth) for accuracy query')
args = parser.parse_args()

# Setup
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# Dataset configuration
dataset = args.dataset

# Load data
if dataset == 'cifar10':
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    # NOTE: Place CIFAR-10 data in 'data/cifar10' directory
    trainset = datasets.CIFAR10(root='data/cifar10', train=True, download=True, transform=transform)
elif dataset == 'cifar100':
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4866, 0.4409), (0.2009, 0.1984, 0.2023)),
    ])
    trainset = datasets.CIFAR100(root='data/cifar100', train=True, download=True, transform=transform)
elif dataset == 'ImageNet16-120':
    from utilsfold.imagenet16 import ImageNet16
    mean = [x / 255 for x in [122.68, 116.66, 104.01]]
    std = [x / 255 for x in [63.22, 61.26, 65.09]]
    transform = transforms.Compose([
        transforms.RandomCrop(16, padding=2),
        transforms.Resize(16),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    trainset = ImageNet16('data/ImageNet16', True, transform, 120)

train_loader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True, num_workers=0, drop_last=True)

# Proxy to use
proxy_name = args.proxy
print(f"Using proxy: {proxy_name}")
print(f"Dataset: {dataset}")

# Initialize Local Search
max_evals = args.max_evals
ls = LocalSearchNAS201Proxy(proxy_name=proxy_name, max_evaluations=max_evals, dataset=dataset, device=device, debug=True)

print(f"Starting Local Search with {proxy_name} proxy...")
start_time = time.time()

# Run search
results = ls.search(train_loader, num_searches=10)

elapsed = time.time() - start_time
print(f"\nSearch completed in {elapsed:.2f}s")
print(f"Total evaluations: {ls.evaluations}")
print(f"Found {len(results)} architectures")

# Query ground-truth accuracy if API path is provided
if args.api_path and os.path.exists(args.api_path):
    print(f"\nQuerying ground-truth accuracy from NAS-Bench-201 API...")
    from nas_201_api import NASBench201API as API
    from easydict import EasyDict as edict
    api = API(args.api_path)
    for r in results:
        arch_str = r['arch_str']
        index = api.query_index_by_arch(arch_str)
        if index >= 0:
            information = api.arch2infos_full[index]
            if dataset == 'cifar10':
                test_info = information.get_metrics(dataset, 'ori-test')
            else:
                test_info = information.get_metrics(dataset, 'x-test')
            r['test_accuracy'] = test_info['accuracy']
            print(f"  Arch: {arch_str}  ->  Test Acc: {test_info['accuracy']:.2f}%")
        else:
            r['test_accuracy'] = None
            print(f"  Arch: {arch_str}  ->  Not found in API")

# Save results
output = {
    'proxy': proxy_name,
    'dataset': dataset,
    'results': results,
    'total_evaluations': ls.evaluations,
    'time': elapsed
}

output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results', 'LocalSearch-nas201')
os.makedirs(output_dir, exist_ok=True)
output_file = os.path.join(output_dir, f'ls_results_{dataset}_{proxy_name}.json')

with open(output_file, 'w') as f:
    json.dump(output, f, indent=2, default=lambda x: x.tolist() if hasattr(x, 'tolist') else x)

print(f"\nResults saved to {output_file}")
