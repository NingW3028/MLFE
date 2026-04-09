"""
MacroNAS Local Search for NAS-101
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # Specify GPU
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import argparse
import pickle
import random
import numpy as np
import torch


from local_search_proxy import LocalSearchNAS101Proxy
import torchvision.transforms as transforms
import torchvision.datasets as datasets


def query_accuracy_from_cache(matrix, ops, cache):
    """Query architecture validation accuracy from cache"""
    try:
        key = (tuple(map(tuple, matrix)), tuple(ops))
        if key in cache:
            return cache[key]['accuracy']
        return None
    except:
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--proxy', type=str, required=True)
    parser.add_argument('--max_evals', type=int, default=500)
    parser.add_argument('--num_searches', type=int, default=20)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--cache_path', type=str, default='nas101_cache.pkl')
    args = parser.parse_args()
    
    # Set seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    print(f"Proxy: {args.proxy}")
    print(f"Max evals: {args.max_evals}")
    print(f"Seed: {args.seed}")
    
    # Load cache
    print(f"Loading NAS-101 cache from {args.cache_path}...")
    with open(args.cache_path, 'rb') as f:
        nas101_cache = pickle.load(f)
    print(f"Loaded {len(nas101_cache)} architectures")
    
    # Load data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    trainset = datasets.CIFAR10(root='data/cifar10', train=True, download=False, transform=transform)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True, num_workers=0, drop_last=True)
    
    # Run search
    ls = LocalSearchNAS101Proxy(proxy_name=args.proxy, max_evaluations=args.max_evals, device=device, debug=False)
    results = ls.search(train_loader, num_searches=args.num_searches)
    
    # Query accuracies from cache
    print("Querying accuracies from cache...")
    for res in results:
        acc = query_accuracy_from_cache(res['matrix'], res['ops'], nas101_cache)
        res['accuracy'] = acc
        if acc:
            print(f"  Proxy: {res['proxy_score']:.2e}, Accuracy: {acc:.2f}")
    
    # Find best by proxy score
    if results:
        best = max(results, key=lambda x: x['proxy_score'])
        output_data = {
            'best_arch': best['matrix'],
            'best_ops': best['ops'],
            'best_proxy_score': best['proxy_score'],
            'best_accuracy': best.get('accuracy'),
            'all_results': results,
            'proxy': args.proxy,
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
