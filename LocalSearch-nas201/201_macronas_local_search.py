"""
MacroNAS Local Search for NAS-Bench-201
Supports CIFAR-10, CIFAR-100, and ImageNet16-120 datasets.
Optionally queries ground truth accuracy from NAS-Bench-201 API.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import argparse
import pickle
import random
import numpy as np
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from local_search_proxy import LocalSearchNAS201Proxy
from utilsfold.imagenet16 import ImageNet16


def get_dataloader(dataset, data_root='data', batch_size=32):
    """Get data loader for specified dataset"""
    if dataset == 'cifar10':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        trainset = datasets.CIFAR10(
            root=os.path.join(data_root, 'cifar10'), train=True, download=False, transform=transform)

    elif dataset == 'cifar100':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4866, 0.4409), (0.2009, 0.1984, 0.2023))
        ])
        trainset = datasets.CIFAR100(
            root=os.path.join(data_root, 'cifar100'), train=True, download=False, transform=transform)

    elif dataset == 'ImageNet16-120':
        mean = [x / 255 for x in [122.68, 116.66, 104.01]]
        std = [x / 255 for x in [63.22, 61.26, 65.09]]
        transform = transforms.Compose([
            transforms.RandomCrop(16, padding=2),
            transforms.Resize(16),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        trainset = ImageNet16(os.path.join(data_root, 'ImageNet16'), True, transform, 120)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    train_loader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)
    return train_loader


def query_accuracy_from_api(arch_str, dataset, api):
    """Query architecture accuracy from NAS-Bench-201 API"""
    if api is None:
        return None
    try:
        arch_index = api.query_index_by_arch(arch_str)
        if arch_index < 0:
            return None

        information = api.arch2infos_full[arch_index]
        if dataset == 'cifar10':
            test_info = information.get_metrics(dataset, 'ori-test')
        else:
            test_info = information.get_metrics(dataset, 'x-test')

        return test_info['accuracy']
    except Exception as e:
        print(f"Warning: Failed to query accuracy: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description='MacroNAS Local Search for NAS-Bench-201')
    parser.add_argument('--proxy', type=str, required=True,
                        help='Proxy name')
    parser.add_argument('--dataset', type=str, default='cifar10',
                        choices=['cifar10', 'cifar100', 'ImageNet16-120'],
                        help='Dataset to use (default: cifar10)')
    parser.add_argument('--max_evals', type=int, default=500,
                        help='Maximum evaluations per search (default: 500)')
    parser.add_argument('--num_searches', type=int, default=20,
                        help='Number of local searches (default: 20)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    parser.add_argument('--output', type=str, required=True,
                        help='Output pickle file path')
    parser.add_argument('--data_root', type=str, default='data',
                        help='Root directory for datasets (default: data)')
    parser.add_argument('--api_path', type=str, default=None,
                        help='Path to NAS-Bench-201 API file (optional, for ground truth accuracy)')
    args = parser.parse_args()

    # Set seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    print(f"Proxy: {args.proxy}")
    print(f"Dataset: {args.dataset}")
    print(f"Max evals: {args.max_evals}")
    print(f"Seed: {args.seed}")

    # Load NAS-201 API if provided
    api = None
    if args.api_path and os.path.exists(args.api_path):
        try:
            from nas_201_api import NASBench201API as API
            print(f"Loading NAS-Bench-201 API from {args.api_path}...")
            api = API(args.api_path, verbose=False)
            print(f"Loaded NAS-Bench-201 API with {len(api)} architectures")
        except Exception as e:
            print(f"Warning: Failed to load NAS-201 API: {e}")
            print("Accuracy query will be disabled")

    # Load data
    print(f"Loading {args.dataset} dataset...")
    train_loader = get_dataloader(args.dataset, args.data_root)

    # Run search
    ls = LocalSearchNAS201Proxy(
        proxy_name=args.proxy,
        max_evaluations=args.max_evals,
        dataset=args.dataset,
        device=device,
        debug=False
    )
    results = ls.search(train_loader, num_searches=args.num_searches)

    # Query accuracies if API available
    if api is not None:
        print("Querying accuracies from NAS-Bench-201 API...")
        for res in results:
            acc = query_accuracy_from_api(res['arch_str'], args.dataset, api)
            res['accuracy'] = acc
            if acc is not None:
                print(f"  Proxy: {res['proxy_score']:.2e}, Accuracy: {acc:.2f}%")

    # Find best by proxy score
    if results:
        best = max(results, key=lambda x: x['proxy_score'])
        output_data = {
            'best_genotype': best['genotype'],
            'best_arch_str': best['arch_str'],
            'best_proxy_score': best['proxy_score'],
            'best_params': best['params'],
            'best_accuracy': best.get('accuracy'),
            'all_results': results,
            'proxy': args.proxy,
            'dataset': args.dataset,
            'seed': args.seed
        }
    else:
        output_data = {'error': 'No results', 'proxy': args.proxy, 'dataset': args.dataset}

    # Save
    with open(args.output, 'wb') as f:
        pickle.dump(output_data, f)

    print(f"Results saved to {args.output}")


if __name__ == '__main__':
    main()
